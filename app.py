import os

import gradio as gr
import openai
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def summarize_text(user_prompt, system_prompt=None):
    """Summarize text using OpenAI's GPT-3 API.

    Args:
        user_prompt (str): Text to be summarized.
        system_prompt (str): Instructions for summarizing text.

    Returns:
        prompt_and_response (dict): Dictionary containing the prompt and response.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    prompt_and_response = {
        "prompt": user_prompt,
        "response": response.choices[0].message.content,
    }

    return prompt_and_response


def llm_response(message):
    # Create system prompt for LLM behavior
    instructions = "Summarize this text. Make concise."

    # Get prompt & response from Summarize function
    prompt_and_response = summarize_text(message, instructions)

    response = prompt_and_response["response"]

    return response


iface = gr.Interface(
    fn=llm_response,
    inputs=[
        "text",
    ],
    outputs="text",
    live=False,
)

iface.launch(debug=True)
