from functools import partial
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
import streamlit as st


@st.cache
def get_data(filename):
    f = pd.read_csv(f"{filename}")
    return f


def local_css(file_name):
    with open(file_name) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


# @st.cache
# def get_model(model):
#     with open(f"models/{model}", "rb") as f:
#         model = pickle.load(f)
#     return model


st.header("**Search faculty at Georgetown Law**")
st.subheader("To start, complete a sentence:")
st.markdown("*I'm looking for a professor who...*")
st.markdown("*Next summer, I want to...*")
st.markdown("*I want to research...*")


faculty = get_data("full_faculty.csv")
tokenized = get_data("tokenized.csv")
tokenized.rename({"Unnamed: 0": "Index"}, axis=1, inplace=True)

request = st.text_input("", "is an expert on gender justice and inequality")
st.markdown("")

my_stop_words = text.ENGLISH_STOP_WORDS


@st.cache
def to_find_your_faculty():
    text_content = tokenized["Value"].str.lower()
    vector = TfidfVectorizer(
        max_df=0.3,
        stop_words=my_stop_words,
        ngram_range=(1, 3),
        lowercase=True,
        use_idf=True,
        norm="l2",
        smooth_idf=True,
    )
    tfidf = vector.fit_transform(text_content)
    return vector, tfidf


vector, tfidf = to_find_your_faculty()


def get_field(name, field):
    return faculty.loc[faculty["Name"] == name, field].values[0]


def search(tfidf_matrix, model, request, df, top_n=5):
    request_transform = model.transform([request])
    similarity = np.dot(request_transform, np.transpose(tfidf_matrix))
    x = np.array(similarity.toarray()[0])
    indices = np.argsort(x)[::-1]
    from_df = df.loc[indices]
    from_df = from_df.drop_duplicates(subset="Name", keep="first").head().values

    for i, (index, highlight, name) in enumerate(from_df, 1):
        partial_field = partial(get_field, name)
        st.markdown(f"**Recommendation #{i}:** {name}")

        email = partial_field("Email")
        st.title_container = st.beta_container()
        col1, mid, col2 = st.beta_columns([1, 3.5, 20])

        img = partial_field("Image")

        if "htt" not in str(img):
            img = "https://pbs.twimg.com/profile_images/1100451034827276290/4puQzDNA_400x400.png"

        with col1:
            st.image(img, width=100)
        with col2:
            st.markdown(f"{highlight}")

            if "edu" in str(email):
                st.markdown(
                    f"Find out more ➔ [Full Bio]({partial_field('URL')}) | [Email](mailto:{partial_field('Email')})"
                )
            else:
                st.markdown(f"Find out more ➔ [Full Bio]({partial_field('URL')})")

        st.markdown("")


search(tfidf, vector, request, tokenized, top_n=5)

local_css("style.css")


# t = "<div>Hello there my <span class='highlight blue'>name <span class='bold'>yo</span> </span> is <span class='highlight grey'>Fanilo <span class='bold'>Name</span></span></div>"
# st.markdown(t, unsafe_allow_html=True)
