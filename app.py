from flask import Flask, request, jsonify
import werkzeug
from posture_image import showimage1 as test

app = Flask(__name__)

@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST" :
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        print("\nReceived image File name : " + imagefile.filename)
        imagefile.save("./uploadedimages/" + filename)
        im,degre = test('./uploadedimages/' + filename)
        degre = str(degre)
    if (im == 1):
	    sonuc = 'Kambur Durus' 
    elif (im == -1):
        sonuc = 'Uzanmis Durus' 
    else:
	    sonuc =  'Dik Durus'

        
    return jsonify({
        "message": sonuc,
        "message1": degre,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)