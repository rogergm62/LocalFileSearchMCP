import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import os
from litellm import acompletion

# Configure how to start our MCP server
server_params = StdioServerParameters(
    command="uv", args=["run", "server.py"])

async def main():

    # Start the *** SERVER *** and establish stdio communication
    async with stdio_client(server_params) as (read, write):

        # Create an MCP ** CLIENT ** session
        async with ClientSession(read, write) as session:

            # Initialize the MCP connection ** CLIENT - SERVER **
            await session.initialize()

            print("Connected to MCP server!")
            print('*'*100)

            # Ask the server which tools are available
            tools = await session.list_tools()
            print(f"This is the server response with the avail tools\n {tools.model_dump_json(indent=2)}")
            input("Enter to continue")
            # Convert MCP tool definitions to LLM-compatible format
            llm_tools = []

            for tool in tools.tools:
                llm_tools.append({"type": "function", "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema}})

            # Display the extracted tool definitions
            for tool in llm_tools:
                print(f"Here is the information about tools extracted from tools\n {json.dumps(tool, indent=2)}")
                print('*'*100)
                input('Enter to continue')
            # Prepare our question for the LLM
            messages = [{
                "role": "user",
                "content": "List all the files and find which ones contain the word Python."
                }]

            # Maximum number of LLM requests
            max_rounds = 5
            for round_number in range(max_rounds):
                print(f"\nAgent Round: {round_number + 1}")

                # Send the question and available tools to the LLM
                response = await acompletion(
                    model="openai/gpt-4.1-mini",
                    api_key=os.environ["OPENAI_API_KEY"],
                    messages=messages,
                    tools=llm_tools,
                    tool_choice="auto")
            
                print(f"Here is the full content of the LLM response:\n{response.model_dump_json(indent=2)}")
                print('*'*100)
                input('Enter to continue')

                # Inspect the LLM's response
                message = response.choices[0].message

                 # If the LLM doesn't request tools, finish
                if not message.tool_calls:
                    print("\nFinal AI Answer:")
                    print(message.content)
                    break
                # Save the assistant's tool requests
                messages.append(message.model_dump(exclude_none=True))

                # Execute all requested MCP tools
                for tool_call in message.tool_calls:
                    # Extract the tool name
                    tool_name = tool_call.function.name

                    # Convert the JSON arguments into a Python dictionary
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"\nCalling MCP tool: {tool_name}")
                    print(f"Arguments: {tool_args}")

                    # Execute the tool through our MCP session
                    result = await session.call_tool(
                        name=tool_name,
                        arguments=tool_args)

                    # Display the result returned by the MCP server
                    print('*'*100)
                    print("\nMCP Tool Result (Executes the tool (tool name and the arguments)):")
                    print(result.model_dump_json(indent=2))
                    
                    # Add each MCP tool result to the conversation
                    # (For this first test, we use one tool call.)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.model_dump_json()})
                else:
                    print("\nMaximum number of agent rounds reached.")
                    print("The agent stopped without generating a final answer.")

                # Ask the LLM to generate its final answer
                final_response = await acompletion(
                    model="openai/gpt-4.1-mini",
                    api_key=os.environ["OPENAI_API_KEY"],
                    messages=messages)

                print("\nFinal AI Answer:")
                print(final_response.choices[0].message.content)

            else:
                print("\nFinal AI Answer:")
                print(message.content)

if __name__ == "__main__":
    asyncio.run(main())
