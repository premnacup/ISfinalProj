import streamlit as st

pages = {
    "Developing": [
        st.Page("pages/page1.py", title="Dataset"),
        st.Page("pages/page2.py", title="Developing the model"),
    ],
    "Demo": [
        st.Page("pages/page3.py", title="Machine Learning"),
        st.Page("pages/page4.py", title="Neural Network"),
    ],
}

pg = st.navigation(pages, position="sidebar")
pg.run()
