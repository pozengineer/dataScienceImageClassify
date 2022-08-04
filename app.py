from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
labelDict= {0: 'colon_aca', 1: 'colon_n', 2: 'lung_aca', 3: 'lung_n', 4: 'lung_scc'}

model= load_model('model.h5')
model.make_predict_function()

def predict_label(x):
    i= image.load_img(x, target_size= (96, 96))
    i= image.img_to_array(i)/ 255.0
    i= i.reshape(1, 96, 96, 3)
    p= model.predict(i).argmax(axis= 1)
    return labelDict[p[0]]
    
@app.route('/', methods= ['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/about')
def about_page():
    return 'Capstone Project: Image Classification Web App'

@app.route('/predict' , methods= ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img= request.files['my_image']
        img_path= 'static/' + img.filename
        img.save(img_path)
        
        p= predict_label(img_path)
        
    return render_template('index.html', prediction= p, img_path= img_path)


if __name__ == "__main__":
    app.run(debug=True)