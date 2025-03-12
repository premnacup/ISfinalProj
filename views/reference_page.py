import streamlit as st

st.title("Reference")
st.divider()
st.write(
    """
    <p style="text-align: justify;">
    <h2>Streamlit Guide</h2>
    <ul>
        <li><a href="https://streamlit.io/">Streamlit</a></li>
    <ul>
    <h2>Dataset</h2>
    <ul>
        <li><a href="https://www.kaggle.com/datasets">kaggle.com</a></li>
    <ul>
    <h2>Information</h2>
    <ul>
        <li><a href="https://www.geeksforgeeks.org/support-vector-machine-algorithm/">SVM</a></li>
        <li><a href="https://www.geeksforgeeks.org/decision-tree-algorithm/">Decision Tree</a></li>
        <li><a href="https://www.geeksforgeeks.org/feedforward-neural-network/">Feedforward Neural Network</a></li>
    </ul>
    <h2>Media</h2>
    <ul>
        <li><a href="https://media.geeksforgeeks.org/wp-content/uploads/20231109124312/Hinge-loss-(2).png">SVM visualization</a></li>
        <li><a href="https://media.geeksforgeeks.org/wp-content/uploads/20250107141217134593/Decision-Tree.webp">Decision Tree visualization</a></li>
        <li><a href="https://media.geeksforgeeks.org/wp-content/uploads/20240601001059/FNN-768.jpg">FNN visualization</a></li>
    </ul>
    </p>
    """,
    unsafe_allow_html=True,
)
