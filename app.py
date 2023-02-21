import os
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        degree = request.form["degree"]

        # single degree embedding setup
        # df = pd.read_csv('processed/' + degree + '/embeddings.csv', index_col=0)

        # cross degrees embedding setup
        df = pd.read_csv('processed/embeddings.csv', index_col=0)

        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
        response = answer_question(df, question=f"""
        Extract the following entities for {degree}, and add new line after each entity. 
        - Degree name:
        - School name:
        - Tuition:<total tuition>
        - Duration: <length of the program>
        - Financial aid available: <Yes or No, if the question can't be answered based on the context, say No>
        - Potential careers: <comma-separated list of career opportunities mentioned>
        """)

        return redirect(url_for("index", result=response))

    result = request.args.get("result")
    print(result)
    return render_template("index.html", result=result)


@app.route("/ama", methods=("GET", "POST"))
def ama():
    if request.method == "POST":
        question = request.form["ama"]
        df = pd.read_csv('processed/embeddings.csv', index_col=0)
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
        response = answer_question(df, question=question)

        return redirect(url_for("index", result=response))

    result = request.args.get("result")
    print(result)
    return render_template("index.html", result=result)


def create_context(
        question, df, max_len=3000, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
        df,
        model="text-davinci-003",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=3000,
        size="ada",
        debug=False,
        max_tokens=500,
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
