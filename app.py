import streamlit as st

pages = {
    "Developing": [
        st.Page("views/dataset_page.py", title="Dataset"),
        st.Page("views/develop_page.py", title="Developing the model"),
    ],
    "Demo": [
        st.Page("views/MLdemo_page.py", title="Machine Learning"),
        st.Page("views/NNdemo_page.py", title="Neural Network"),
    ],
}

pg = st.navigation(pages, position="sidebar")
pg.run()
