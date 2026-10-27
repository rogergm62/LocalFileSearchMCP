import asyncio
import json
import os

from litellm import acompletion
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Tell the client how to start our local MCP server.
server_params = StdioServerParameters(
    command="uv",
    args=["run", "server.py"],
)


def validate_arguments(tool_name: str, arguments: dict) -> bool:
    """Check the expected parameters for our three simple MCP tools."""
    if not isinstance(arguments, dict):
        return False

    if tool_name == "list_files":
        return arguments == {}

    if tool_name in ("search_files", "search_content"):
        return (
            set(arguments.keys()) == {"keyword"}
            and isinstance(arguments["keyword"], str)
            and bool(arguments["keyword"].strip())
        )

    return False


async def main():
    # Open communication streams to the server (starts server.py automatically).
    async with stdio_client(server_params) as (read, write):
        # Set up the MCP protocol session.
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP server!")
            print("*" * 70)

            # Ask the server which tools are available.
            tools = await session.list_tools()
            print("MCP server tool definitions:")
            print(tools.model_dump_json(indent=2))
            # input("Press Enter to continue...")

            # Allow only the tools discovered from this server.
            allowed_tools = {tool.name for tool in tools.tools}
            print(f"Allowed MCP tools: {allowed_tools}")

            # Convert MCP definitions into LiteLLM-compatible tool definitions.
            llm_tools = []
            for tool in tools.tools:
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                })

            for tool in llm_tools:
                print("\nTool definition for the LLM:")
                print(json.dumps(tool, indent=2))
                # input("Press Enter to continue...")

            # Start with one question; the list grows as tool responses arrive.


            # Create an empty conversation history
            messages = []

            # Keep asking questions until the user types exit
            while True:

                question = input("\nYou: ").strip()

                if question.lower() == "exit":
                    print("Goodbye!")
                    break

                # Ignore empty questions
                if not question:
                    continue

                # Save the user's question
                messages.append({
                    "role": "user",
                    "content": question
                })

                # Process this question using our MCP agent
                max_rounds = 5

                for round_number in range(max_rounds):
                    # Existing agent code goes here

                    print(f"\nAgent Round: {round_number + 1}")

                    try:
                        response = await acompletion(
                            model="openai/gpt-4.1-mini",
                            api_key=os.environ["OPENAI_API_KEY"],
                            messages=messages,
                            tools=llm_tools,
                            tool_choice="auto",
                        )
                    except Exception as e:
                        print(f"LLM request failed: {e}")
                        break

                    print("\n ############## Full LLM response: ###############")
                    print(response.model_dump_json(indent=2))
                    # input("Press Enter to continue...")

                    message = response.choices[0].message

                    # A response without tool calls is the final answer.
                    if not message.tool_calls:
                        print("\nFinal AI Answer:")
                        print(message.content)
                        break

                    # Store all the assistant's tool requests first.
                    messages.append(message.model_dump(exclude_none=True))

                    # Execute EACH tool request and return a result for EACH call ID.
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        print(f"\nRequested MCP tool: {tool_name}")

                        # 1. Check the requested tool is allowed.
                        if tool_name not in allowed_tools:
                            error_message = f"Tool '{tool_name}' is not allowed."
                            print(error_message)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_message,
                            })
                            continue

                        # 2. Parse the JSON arguments safely.
                        try:
                            tool_args = json.loads(tool_call.function.arguments)
                        except (json.JSONDecodeError, TypeError) as e:
                            error_message = f"Invalid JSON arguments for '{tool_name}': {e}"
                            print(error_message)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_message,
                            })
                            continue

                        # 3. Check that the arguments match this project's tools.
                        if not validate_arguments(tool_name, tool_args):
                            error_message = f"Invalid arguments for tool '{tool_name}'."
                            print(error_message)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_message,
                            })
                            continue

                        print(f"Arguments: {tool_args}")

                        # 4. Ask the MCP server to execute the tool.
                        try:
                            result = await session.call_tool(
                                name=tool_name,
                                arguments=tool_args,
                            )
                        except Exception as e:
                            error_message = f"MCP tool execution failed: {e}"
                            print(error_message)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_message,
                            })
                            continue

                        # 5. Inspect the MCP response, including tool-reported errors.
                        if result.is_error:
                            print("Warning: MCP tool reported an error.")
                        else:
                            print("MCP tool executed successfully.")

                        print("MCP Tool Result:")
                        print(result.model_dump_json(indent=2))

                        # 6. Save the tool result so the LLM can use it next round.
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.model_dump_json(),
                        })

                    # Next round: the LLM sees the updated messages list.
                else:
                    print("\nMaximum number of agent rounds reached.")
                    print("The agent stopped without generating a final answer.")


if __name__ == "__main__":
    asyncio.run(main())
