import streamlit as st





#GUI
st.title("Tablero Empreafdefewfsas🎈")
st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")






st.markdown("*Streamlit* is **really** ***cool***.")
st.markdown('''
    :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
    :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')
st.markdown("Here's a bouquet &mdash;\
            :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

multi = '''If you end a line with two spaces,
a soft return is used for the next line.

Two (or more) newline characters in a row will result in a hard return.
'''
st.markdown(multi)

st.code(f"""
import streamlit as st

st.markdown('''''')
""")



#x = st.slider('x')  # 👈 this is a widget
#st.write(x, 'squared is', x * x)

#tabla1=st.dataframe(vtiger)


#st.text_input("Your name", key="name")
#st.session_state.name

#https://docs.streamlit.io/

#option = st.selectbox(   'Which number do you like best?',df['first column']) #filtro seleccion individual despegable
#add_slider = st.sidebar.slider('Select a range of values',0.0, 100.0, (25.0, 75.0)) #filtro barra


#chosen = st.radio('Sorting hat',("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin")) #visible filtro selecion individual







# streamlit run C:\Users\Usuario\Desktop\Streamlit\main.py


