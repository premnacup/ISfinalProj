import streamlit as st

pages = {
    "Developing": [
        st.Page("page1.py", title="Dataset"),
        st.Page("page2.py", title="Developing the model"),
    ],
    "Demo": [
        st.Page("page3.py", title="Machine Learning"),
        st.Page("page4.py", title="Neural Network"),
    ],
}

pg = st.navigation(pages, position="sidebar")
pg.run()
