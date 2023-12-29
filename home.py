import streamlit as st
tab1, tab2 = st.tabs(["Home", "About"])
with tab1:
    st.title("Classifier")
    import streamlit as st
    from PIL import Image
    from ultralytics import YOLO

    @st.cache_resource
    def load_model(): 
        mod = YOLO("best.pt")
        return mod
    img = st.file_uploader('Select the image', type=['jpg','png','jpeg'])
    if img is not None:
        img = Image.open(img)
        st.markdown('Image Visualization')
        st.image(img)
        st.header('Moth Classification')
        model = load_model()
        res = model.predict(img)
        label = res[0].probs.top5
        conf = res[0].probs.top5conf
        conf = conf.tolist()
        col1,col2 = st.columns(2)
        col1.subheader(res[0].names[label[0]].title() +' with '+ str(conf[0])+' Confidence')
        col2.subheader(res[0].names[label[1]].title() +' with '+ str(conf[1])+' Confidence')
with tab2:
    st.title("About Me")