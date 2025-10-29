import streamlit as st
from openai import OpenAI

from src.config.settings import settings
from src.core.llm_client import LLMClient
from src.utils.detect_ingredients import detect_ingredients
from src.utils.detect_user_type import detect_user_type
from src.utils.get_meal_time import get_meal_time_from_hour

st.title("ChatGPT-like clone")

client = OpenAI(
    base_url=settings.OPENAI_BASE_URL,
    api_key=settings.OPENAI_API_KEY,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 
        if prompt.startswith("Hôm nay ăn gì?"):
            # Define the prompt template
            prompt_template = "Recommend foods based on user's preferences"

            # Serving type: personal | family | none
            serving_type = detect_user_type(LLMClient(), {"message": st.session_state.messages})
            prompt_template += "\n- Serving type: " + serving_type
            if serving_type == "none":
                # TODO: Required
                st.session_state.messages.append({"role": "assistant", "content": "Person unknow"})

            # Extract ingredients from the user input
            ingredients = detect_ingredients(st.session_state.messages)
            if ingredients:
                prompt_template += "\n- Ingredients: " + ingredients

            # Meal type by time
            meal_type = get_meal_time_from_hour()
            if meal_type:
                prompt_template += "\n- Meal type: " + meal_type

            # Append the system prompt
            st.session_state.messages.append({"role": "system", "content": prompt_template})
            print(st.session_state.messages)
        # elif: "food details"
        # # TODO:
        # else:
        #     ""

        stream = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=st.session_state.messages,
            temperature=settings.OPENAI_TEMPERATURE,
            max_completion_tokens=settings.OPENAI_MAX_TOKENS,
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
