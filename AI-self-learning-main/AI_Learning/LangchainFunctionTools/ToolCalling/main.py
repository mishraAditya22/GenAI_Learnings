import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from ll_config import AzureOpenAIConfig as LLMConfig
import json
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
modelName = os.getenv("DOCUMENT_MODEL") 
if not modelName:
    print("Model name not specified !!")
    exit(1)


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

# Example dummy function hard coded to return the same user details
# In production, this could be your backend API or an external API
def get_my_personal_details(name=None, age="24"):
    """Get my personal details"""
    if not name or name.lower() != "aditya":
        return json.dumps({"error": "Personal details of the user not found."})
    personal_info = {
        "name": name,
        "age": age,
        "location": "New Delhi",
        "hobbies": ["reading", "coding", "traveling"],
    }
    return json.dumps(personal_info)



tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location, example 'what is the weather in San Francisco, CA?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name":"get_my_personal_details",
            "description": "Use this tool to get details about user , example give me my personal details or what do you know about me ",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Your name",
                    },
                    "age": {
                        "type": "string",
                        "description": "Your age",
                    },
                },
                "required": ["name"],
            },
        }
    }
]

system_prompt = """You are a helpful assistant. If you receive tool results, you must use them as the source of truth and share the information with the user. Do not refuse to answer if a tool result is available."""

def chatWithUser(user_input, history=None):
    """
    Function to chat with the user and call tools based on the input.
    """
    llm_config = LLMConfig()
    client = llm_config.get_client()

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=modelName,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # Automatically choose the best tool based on the input
        max_tokens=1000,
        temperature=0.7,
    )

    print(f"Response: {response.choices[0].message}")

    tool_calls = getattr(response.choices[0].message, "tool_calls", None)
    if tool_calls:
        tool_call = tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        if function_name == "get_current_weather":
            result = get_current_weather(**arguments)
            weather = json.loads(result)
            # Format the weather information as a readable message
            tool_result_json = f"""Here is the current weather information:
                Location: {weather["location"]}
                Temperature: {weather["temperature"]} {weather["unit"]}
                Forecast: {weather["forecast"]}"""
        elif function_name == "get_my_personal_details":
            print("Tool call arguments:", arguments)  # Debug line
            result = get_my_personal_details(**arguments)
            info = json.loads(result)
            if "error" in info:
                tool_result_json = f"Sorry, {info['error']}"
            else:
                tool_result_json = f"""Here are some details of yours :
                    Name: {info.get('name')}
                    Age: {info.get('age')}
                    Location: {info.get('location')}
                    Hobbies: {', '.join(info.get('hobbies', []))}"""
        else:
            tool_result_json = json.dumps({
                "id": tool_call.id,
                "type": "unknown_tool",
                "message": "Tool result: Unknown tool."
            })

        # Add the tool result as an assistant message
        # messages.append({"role": "assistant", "content": tool_result_json})

        # # Send updated messages back to LLM for a natural response
        # final_response = client.chat.completions.create(
        #     model=modelName,
        #     messages=messages,
        #     max_tokens=1000,
        #     temperature=0.7,
        # )
        # return final_response.choices[0].message.content if final_response.choices[0].message.content is not None else tool_result_json

        return tool_result_json
    # If no tool call, return the assistant's message content
    return response.choices[0].message.content if response.choices[0].message.content is not None else ""

def gradio_chat(user_input, history=[]):
    """
    Gradio interface function. Passes user input and history to chatWithUser, then allows sending response back to LLM.
    """
    response = chatWithUser(user_input, history)
    history = history + [[user_input, response]]
    # Build messages in Gradio format
    messages = []
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages, history


with gr.Blocks() as demo:
    gr.Markdown("# Chat with LLM and Function Calling")
    chatbot = gr.Chatbot(type="messages")
    with gr.Row():
        user_input = gr.Textbox(label="Your message")
        send_btn = gr.Button("Send")
    state = gr.State([])

    def send_message(user_input, history):
        messages, history = gradio_chat(user_input, history)
        return messages, history

    send_btn.click(send_message, inputs=[user_input, state], outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch()