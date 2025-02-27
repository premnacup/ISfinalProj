import streamlit as st

pages = {
    "Developing": [
        st.Page("views/page1.py", title="Dataset"),
        st.Page("views/page2.py", title="Developing the model"),
    ],
    "Demo": [
        st.Page("views/page3.py", title="Machine Learning"),
        st.Page("views/page4.py", title="Neural Network"),
    ],
}

pg = st.navigation(pages, position="sidebar")
pg.run()
