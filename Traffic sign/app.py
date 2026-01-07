from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from gtts import gTTS
import base64
import tempfile
from datetime import datetime

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load the trained model
def load_model():
    try:
        model = tf.keras.models.load_model('model/traffic_sign_model.h5')
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model at startup
model = load_model()

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'gtts_lang': 'en'},
    'es': {'name': 'Spanish', 'gtts_lang': 'es'},
    'fr': {'name': 'French', 'gtts_lang': 'fr'},
    'sa': {'name': 'Sanskrit', 'gtts_lang': 'sa'},
    'pa': {'name': 'Punjabi', 'gtts_lang': 'pa'},
    'hi': {'name': 'Hindi', 'gtts_lang': 'hi'},
    'ja': {'name': 'Japanese', 'gtts_lang': 'ja'},
    'zh': {'name': 'Chinese', 'gtts_lang': 'zh'},
    'ta': {'name': 'Tamil', 'gtts_lang': 'ta'},
    'ar': {'name': 'Arabic', 'gtts_lang': 'ar'},
    'kn': {'name': 'Kannada', 'gtts_lang': 'kn'} 
}
classes = {
    0: {
        "name": "Speed limit 20",
        "guidance": {
            'en': "Reduce your speed to 20 kilometers per hour. This is typically found in residential areas or school zones.",
            'es': "Reduzca su velocidad a 20 kilómetros por hora. Esto se encuentra típicamente en zonas residenciales o escolares.",
            'fr': "Réduisez votre vitesse à 20 kilomètres par heure. On trouve généralement cela dans les zones résidentielles ou scolaires.",
            'hi': "अपनी गति 20 किलोमीटर प्रति घंटे तक कम करें। यह आमतौर पर आवासीय क्षेत्रों या स्कूल क्षेत्रों में पाया जाता है।",
            'ja': "速度を時速20キロに減速してください。これは通常、住宅地や学校区域で見られます。",
            'kn': "ನಿಮ್ಮ ವೇಗವನ್ನು ಗಂಟೆಗೆ 20 ಕಿಲೋಮೀಟರ್ಗೆ ಕಡಿಮೆ ಮಾಡಿ. ಇದು ಸಾಮಾನ್ಯವಾಗಿ ನಿವಾಸಿ ಪ್ರದೇಶಗಳಲ್ಲಿ ಅಥವಾ ಶಾಲಾ ವಲಯಗಳಲ್ಲಿ ಕಂಡುಬರುತ್ತದೆ.",
            'sa': "त्वं वेगं प्रतिघण्टायां २० किलोमीटरपर्यन्तं कुरु। एतत् सामान्यतः निवासक्षेत्रेषु विद्यालयक्षेत्रेषु वा दृश्यते।",
            'pa': "ਆਪਣੀ ਰਫ਼ਤਾਰ 20 ਕਿਲੋਮੀਟਰ ਪ੍ਰਤੀ ਘੰਟਾ ਤੱਕ ਘਟਾਓ। ਇਹ ਆਮ ਤੌਰ 'ਤੇ ਰਿਹਾਇਸ਼ੀ ਇਲਾਕਿਆਂ ਜਾਂ ਸਕੂਲ ਖੇਤਰਾਂ ਵਿੱਚ ਮਿਲਦਾ ਹੈ।",
            'ta': "உங்கள் வேகத்தை மணிக்கு 20 கிலோமீட்டராகக் குறைக்கவும். இது பொதுவாக குடியிருப்புப் பகுதிகள் அல்லது பள்ளி மண்டலங்களில் காணப்படுகிறது."
        }
    },
    1: {
        "name": "Speed limit 30", 
        "guidance": {
            'en': "Reduce your speed to 30 kilometers per hour. Common in urban areas with high pedestrian activity.",
            'es': "Reduzca su velocidad a 30 kilómetros por hora. Común en áreas urbanas con alta actividad peatonal.",
            'fr': "Réduisez votre velocidad a 30 kilomètres par heure. Courant dans les zones urbaines à forte activité piétonne.",
            'hi': "अपनी गति 30 किलोमीटर प्रति घंटे तक कम करें। उच्च पैदल यात्री गतिविधि वाले शहरी क्षेत्रों में आम।",
            'ja': "速度を時速30キロに減速してください。歩行者活動の多い都市部でよく見られます。",
            'kn': "ನಿಮ್ಮ ವೇಗವನ್ನು ಗಂಟೆಗೆ 30 ಕಿಲೋಮೀಟರ್ಗೆ ಕಡಿಮೆ ಮಾಡಿ. ಹೆಚ್ಚು ಪಾದಚಾರಿ ಚಟುವಟಿಕೆಯಿರುವ ನಗರ ಪ್ರದೇಶಗಳಲ್ಲಿ ಸಾಮಾನ್ಯ.",
            'sa': "त्वं वेगं प्रतिघण्टायां ३० किलोमीटरपर्यन्तं कुरु। उच्चपादचारिसक्रियायुक्तनगरक्षेत्रेषु सामान्यम्।",
            'pa': "ਆਪਣੀ ਰਫ਼ਤਾਰ 30 ਕਿਲੋਮੀਟਰ ਪ੍ਰਤੀ ਘੰਟਾ ਤੱਕ ਘਟਾਓ। ਉੱਚ ਪੈਦਲ ਚਲਣ ਵਾਲੀਆਂ ਗਤੀਵਿਧੀਆਂ ਵਾਲੇ ਸ਼ਹਿਰੀ ਖੇਤਰਾਂ ਵਿੱਚ ਆਮ।",
            'ta': "உங்கள் வேகத்தை மணிக்கு 30 கிலோமீட்டராகக் குறைக்கவும். அதிக நடமாட்டம் உள்ள நகர்ப்புற பகுதிகளில் பொதுவானது."
        }
    },
    2: {
        "name": "Speed limit 50", 
        "guidance": {
            'en': "Reduce your speed to 50 kilometers per hour. Standard speed limit in built-up areas.",
            'es': "Reduzca su velocidad a 50 kilómetros por hora. Límite de velocidad estándar en áreas edificadas.",
            'fr': "Réduisez votre vitesse à 50 kilomètres par heure. Limite de vitesse standard dans les zones bâties.",
            'hi': "अपनी गति 50 किलोमीटर प्रति घंटे तक कम करें। बसे हुए क्षेत्रों में मानक गति सीमा।",
            'ja': "速度を時速50キロに減速してください。市街地での標準速度制限。",
            'kn': "ನಿಮ್ಮ ವೇಗವನ್ನು ಗಂಟೆಗೆ 50 ಕಿಲೋಮೀಟರ್ಗೆ ಕಡಿಮೆ ಮಾಡಿ. ಕಟ್ಟಡಗಳಿರುವ ಪ್ರದೇಶಗಳಲ್ಲಿ ಪ್ರಮಾಣಿತ ವೇಗ ಮಿತಿ.",
            'sa': "त्वं वेगं प्रतिघण्टायां ५० किलोमीटरपर्यन्तं कुरु। निर्मितक्षेत्रेषु मानकवेगसीमा।",
            'pa': "ਆਪਣੀ ਰਫ਼ਤਾਰ 50 ਕਿਲੋਮੀਟਰ ਪ੍ਰਤੀ ਘੰਟਾ ਤੱਕ ਘਟਾਓ। ਬਣੇ ਹੋਏ ਇਲਾਕਿਆਂ ਵਿੱਚ ਮਾਨਕ ਗਤੀ ਸੀਮਾ।",
            'ta': "உங்கள் வேகத்தை மணிக்கு 50 கிலோமீட்டராகக் குறைக்கவும். கட்டமைக்கப்பட்ட பகுதிகளில் நிலையான வேக வரம்பு."
        }
    },
    3: {
        "name": "Speed limit 60", 
        "guidance": {
            'en': "Reduce your speed to 60 kilometers per hour. Often found on rural roads or outskirts of cities.",
            'es': "Reduzca su velocidad a 60 kilómetros por hora. A menudo se encuentra en carreteras rurales o afueras de ciudades.",
            'fr': "Réduisez votre vitesse à 60 kilómetros par heure. Souvent trouvé sur les routes rurales ou à la périphérie des villes.",
            'hi': "अपनी गति 60 किलोमीटर प्रति घंटे तक कम करें। अक्सर ग्रामीण सड़कों या शहरों के बाहरी इलाकों में पाया जाता है।",
            'ja': "速度を時速60キロに減速してください。田舎道や都市の郊外でよく見られます。",
            'kn': "ನಿಮ್ಮ ವೇಗವನ್ನು ಗಂಟೆಗೆ 60 ಕಿಲೋಮೀಟರ್ಗೆ ಕಡಿಮೆ ಮಾಡಿ. ಗ್ರಾಮೀಣ ರಸ್ತೆಗಳಲ್ಲಿ ಅಥವಾ ನಗರಗಳ ಹೊರವಲಯಗಳಲ್ಲಿ ಸಾಮಾನ್ಯವಾಗಿ ಕಂಡುಬರುತ್ತದೆ.",
            'sa': "त्वं वेगं प्रतिघण्टायां ६० किलोमीटरपर्यन्तं कुरु। ग्रामीणमार्गेषु नगराणां उपान्तेषु वा अनेकधा दृश्यते।",
            'pa': "ਆਪਣੀ ਰਫ਼ਤਾਰ 60 ਕਿਲੋਮੀਟਰ ਪ੍ਰਤੀ ਘੰਟਾ ਤੱਕ ਘਟਾਓ। ਅਕਸਰ ਪੇਂਡੂ ਸੜਕਾਂ ਜਾਂ ਸ਼ਹਿਰਾਂ ਦੇ ਬਾਹਰੀ ਇਲਾਕਿਆਂ ਵਿੱਚ ਮਿਲਦਾ ਹੈ।",
            'ta': "உங்கள் வேகத்தை மணிக்கு 60 கிலோமீட்டராகக் குறைக்கவும். பெரும்பாலும் கிராமப்புற சாலைகளில் அல்லது நகரங்களின் புறநகர்ப் பகுதிகளில் காணப்படுகிறது."
        }
    },
    4: {
        "name": "Speed limit 70", 
        "guidance": {
            'en': "Reduce your speed to 70 kilometers per hour. Typically on main roads outside urban areas.",
            'es': "Reduzca su velocidad a 70 kilómetros por hora. Normalmente en carreteras principales fuera de áreas urbanas.",
            'fr': "Réduisez votre vitesse à 70 kilómetros par heure. Typiquement sur les routes principales en dehors des zones urbaines.",
            'hi': "अपनी गति 70 किलोमीटर प्रति घंटे तक कम करें। आमतौर पर शहरी क्षेत्रों के बाहर मुख्य सड़कों पर।",
            'ja': "速度を時速70キロに減速してください。通常、市街地外の幹線道路にあります。",
            'kn': "ನಿಮ್ಮ ವೇಗವನ್ನು ಗಂಟೆಗೆ 70 ಕಿಲೋಮೀಟರ್ಗೆ ಕಡಿಮೆ ಮಾಡಿ. ಸಾಮಾನ್ಯವಾಗಿ ನಗರ ಪ್ರದೇಶಗಳ ಹೊರಗಿನ ಮುಖ್ಯ ರಸ್ತೆಗಳಲ್ಲಿ.",
            'sa': "त्वं वेगं प्रतिघण्टायां ७० किलोमीटरपर्यन्तं कुरु। नगरक्षेत्रबहिः मुख्यमार्गेषु सामान्यतः।",
            'pa': "ਆਪਣੀ ਰਫ਼ਤਾਰ 70 ਕਿਲੋਮੀਟਰ ਪ੍ਰਤੀ ਘੰਟਾ तੱਕ ਘਟਾਓ। ਆਮ ਤੌਰ 'ਤੇ ਸ਼ਹਿਰੀ ਇਲਾਕਿਆਂ ਤੋਂ ਬਾਹਰ ਮੁੱਖ ਸੜਕਾਂ 'ਤੇ।",
            'ta': "உங்கள் வேகத்தை மணிக்கு 70 கிலோமீட்டராகக் குறைக்கவும். பொதுவாக நகர்ப்புற பகுதிகளுக்கு வெளியே உள்ள முக்கிய சாலைகளில்."
        }
    },
    5: {
        "name": "Speed limit 80", 
        "guidance": {
            'en': "Reduce your speed to 80 kilometers per hour. Common on highways and major roads.",
            'es': "Reduzca su velocidad a 80 kilómetros por hora. Común en autopistas y carreteras principales.",
            'fr': "Réduisez votre vitesse à 80 kilómetros par heure. Courant sur les autoroutes et les routes principales.",
            'hi': "अपनी गति 80 किलोमीटर प्रति घंटे तक कम करें। राजमार्गों और प्रमुख सड़कों पर आम।",
            'ja': "速度を時速80キロに減速してください。高速道路や主要道路でよく見られます。",
            'kn': "ನಿಮ್ಮ ವೇಗವನ್ನು ಗಂಟೆಗೆ 80 ಕಿಲೋमೀಟರ್ಗೆ ಕಡಿಮೆ ಮಾಡಿ. ಹೆದ್ದಾರಿಗಳು ಮತ್ತು ಪ್ರಮುಖ ರಸ್ತೆಗಳಲ್ಲಿ ಸಾಮಾನ್ಯ.",
            'sa': "त्वं वेगं प्रतिघण्टायां ८० किलोमीटरपर्यन्तं कुरु। द्रुतमार्गेषु प्रमुखमार्गेषु च सामान्यम्।",
            'pa': "ਆਪਣੀ ਰਫ਼ਤਾਰ 80 ਕਿਲੋਮੀਟਰ ਪ੍ਰਤੀ ਘੰਟਾ ਤੱਕ ਘਟਾਓ। ਹਾਈਵੇਅ ਅਤੇ ਪ੍ਰਮੁੱਖ ਸੜਕਾਂ 'ਤੇ ਆਮ।",
            'ta': "உங்கள் வேகத்தை மணிக்கு 80 கிலோமீட்டராகக் குறைக்கவும். நெடுஞ்சாலைகள் மற்றும் முக்கிய சாலைகளில் பொதுவானது."
        }
    },
    6: {
        "name": "End of speed limit 80", 
        "guidance": {
            'en': "The 80 km/h speed limit has ended. Check for new speed limits ahead.",
            'es': "El límite de velocidad de 80 km/h ha terminado. Verifique los nuevos límites de velocidad más adelante.",
            'fr': "La limite de vitesse de 80 km/h est terminée. Vérifiez les nouvelles limites de vitesse devant vous.",
            'hi': "80 किमी/घंटा की गति सीमा समाप्त हो गई है। आगे नई गति सीमा की जाँच करें।",
            'kn': "ಗಂಟೆಗೆ 80 ಕಿಮೀ ವೇಗ ಮಿತಿ ಮುಗಿದಿದೆ. ಮುಂದೆ ಹೊಸ ವೇಗ ಮಿತಿಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
            'ja': "時速80キロの速度制限が終了しました。前方の新しい速度制限を確認してください。",
            'sa': "८० किमी/घं वेगसीमा समाप्ता। अग्रे नूतनवेगसीमाः परीक्ष्यन्ताम्।",
            'pa': "80 ਕਿਮੀ/ਘੰਟਾ ਦੀ ਗਤੀ ਸੀਮਾ ਖਤਮ ਹੋ ਗਈ ਹੈ। ਅੱਗੇ ਨਵੀਆਂ ਗਤੀ ਸੀਮਾਵਾਂ ਦੀ ਜਾਂਚ ਕਰੋ।",
            'ta': "மணிக்கு 80 கிமீ வேக வரம்பு முடிவடைந்தது. முன்னால் புதிய வேக வரம்புகளைச் சரிபார்க்கவும்."
        }
    },
    7: {
        "name": "Speed limit 100", 
        "guidance": {
            'en': "Reduce your speed to 100 kilometers per hour. Standard speed limit on highways.",
            'es': "Reduzca su velocidad a 100 kilómetros por hora. Límite de velocidad estándar en autopistas.",
            'fr': "Réduisez votre vitesse à 100 kilómetros par heure. Limite de vitesse standard sur les autoroutes.",
            'hi': "अपनी गति 100 किलोमीटर प्रति घंटे तक कम करें। राजमार्गों पर मानक गति सीमा।",
            'kn': "ನಿಮ್ಮ ವೇಗವನ್ನು ಗಂಟೆಗೆ 100 ಕಿಲೋಮೀಟರ್ಗೆ ಕಡಿಮೆ ಮಾಡಿ. ಹೆದ್ದಾರಿಗಳಲ್ಲಿ ಪ್ರಮಾಣಿತ ವೇಗ ಮಿತಿ.",
            'ja': "速度を時速100キロに減速してください。高速道路での標準速度制限。",
            'sa': "त्वं वेगं प्रतिघण्टायां १०० किलोमीटरपर्यन्तं कुरु। द्रुतमार्गेषु मानकवेगसीमा।",
            'pa': "ਆਪਣੀ ਰਫ਼ਤਾਰ 100 ਕਿਲੋਮੀਟਰ ਪ੍ਰਤੀ ਘੰਟਾ ਤੱਕ ਘਟਾਓ। ਹਾਈਵੇਅ 'ਤੇ ਮਾਨਕ ਗਤੀ ਸੀਮਾ।",
            'ta': "உங்கள் வேகத்தை மணிக்கு 100 கிலோமீட்டராகக் குறைக்கவும். நெடுஞ்சாலைகளில் நிலையான வேக வரம்பு."
        }
    },
    8: {
        "name": "Speed limit 120", 
        "guidance": {
            'en': "Reduce your speed to 120 kilometers per hour. Maximum speed limit on German autobahns.",
            'es': "Reduzca su velocidad a 120 kilómetros por hora. Límite máximo de velocidad en las autopistas alemanas.",
            'fr': "Réduisez votre vitesse à 120 kilómetros par heure. Limite de vitesse maximale sur les autoroutes allemandes.",
            'hi': "अपनी गति 120 किलोमीटर प्रति घंटे तक कम करें। जर्मन ऑटोबान पर अधिकतम गति सीमा।",
            'ja': "速度を時速120キロに減速してください。ドイツのアウトバーンでの最高速度制限。",
            'kn': "ನಿಮ್ಮ ವೇಗವನ್ನು ಗಂಟೆಗೆ 120 ಕಿಲೋमೀಟರ್ಗೆ ಕಡಿಮೆ ಮಾಡಿ. ಜರ್ಮನ್ ಆಟೋಬಾನ್ಗಳಲ್ಲಿ ಗರಿಷ್ಠ ವೇಗ ಮಿತಿ.",
            'sa': "त्वं वेगं प्रतिघण्टायां १२० किलोमीटरपर्यन्तं कुरु। जर्मनदेशीयद्रुतमार्गेषु अधिकतमवेगसीमा।",
            'pa': "ਆਪਣੀ ਰਫ਼ਤਾਰ 120 ਕਿਲੋਮੀਟਰ ਪ੍ਰਤੀ ਘੰਟਾ ਤੱਕ ਘਟਾਓ। ਜਰਮਨ ਆਟੋਬਾਨ 'ਤੇ ਅਧਿਕਤਮ ਗਤੀ ਸੀਮਾ।",
            'ta': "உங்கள் வேகத்தை மணிக்கு 120 கிலோமீட்டராகக் குறைக்கவும். ஜெர்மன் ஆட்டோபான்களில் அதிகபட்ச வேக வரம்பு."
        }
    },
    9: {
        "name": "No passing", 
        "guidance": {
            'en': "Do not overtake other vehicles. Passing is prohibited due to limited visibility or road conditions.",
            'es': "No adelante a otros vehículos. Está prohibido adelantar debido a visibilidad limitada o condiciones de la carretera.",
            'fr': "Ne doublez pas les otros véhicules. Le dépassement est interdit en raison d'une visibilité limitée ou des conditions routières.",
            'hi': "दूसरे वाहनों को ओवरटेक न करें। सीमित दृश्यता या सड़क की स्थिति के कारण ओवरटेकिंग प्रतिबंधित है।",
            'ja': "他の車両を追い越さないでください。視界不良や道路状況により追い越しは禁止されています。",
            'kn': "ಇತರ ವಾಹನಗಳನ್ನು ಮುಂದೆ ಹೋಗಬೇಡಿ. ಸೀಮಿತ ದೃಶ್ಯತೆ ಅಥವಾ ರಸ್ತೆ ಪರಿಸ್ಥಿತಿಗಳಿಂದಾಗಿ ಮುಂದೆ ಹೋಗುವುದು ನಿಷೇಧಿಸಲಾಗಿದೆ.",
            'sa': "अन्यवाहनानि अतिक्रम्य न गच्छेत्। सीमितदृश्यतया मार्गपरिस्थितिभिः वा अतिक्रमणं निषिद्धम्।",
            'pa': "ਦੂਸਰੀਆਂ ਗੱਡੀਆਂ ਨੂੰ ਓਵਰਟੇਕ ਨਾ ਕਰੋ। ਸੀਮਿਤ ਦ੍ਰਿਸ਼ਟੀ ਜਾਂ ਸੜਕ ਹਾਲਤਾਂ ਕਾਰਨ ਓਵਰਟੇਕਿੰਗ 'ਤੇ ਪਾਬੰਦੀ ਹੈ।",
            'ta': "பிற வாகனங்களை முந்திச் செல்லாதீர்கள். வரம்புக்குட்பட்ட கண்ணொளி அல்லது சாலை நிலைமைகள் காரணமாக முந்திச் செல்வது தடைசெய்யப்பட்டுள்ளது."
        }
    },
    10: {
        "name": "No passing for vehicles over 3.5 tons", 
        "guidance": {
            'en': "Vehicles over 3.5 tons are not allowed to pass other vehicles. Lighter vehicles may still pass.",
            'es': "Los vehículos de más de 3,5 toneladas no pueden adelantar a otros vehículos. Los vehículos más ligeros aún pueden adelantar.",
            'fr': "Les véhicules de plus de 3,5 tonnes ne sont pas autorisés à dépasser d'autres véhicules. Les véhiculos más ligeros pueden toujours dépasser.",
            'hi': "3.5 टन से अधिक के वाहनों को अन्य वाहनों को ओवरटेक करने की अनुमति नहीं है। हल्के वाहन अभी भी ओवरटेक कर सकते हैं।",
            'ja': "3.5トンを超える車両は他の車両を追い越せません。軽量車両はまだ追い越せます。",
            'kn': "3.5 ಟನ್ಗಳಿಗಿಂತ ಹೆಚ್ಚು ವಾಹನಗಳಿಗೆ ಇತರ ವಾಹನಗಳನ್ನು ಮುಂದೆ ಹೋಗಲು ಅನುಮತಿ ಇಲ್ಲ. ಹಗುರ ವಾಹನಗಳು ಇನ್ನೂ ಮುಂದೆ ಹೋಗಬಹುದು.",
            'sa': "३.५ टनोपरिवाहनानि अन्यवाहनानि अतिक्रम्य न गच्छन्ति। लघुवाहनानि अद्यापि अतिक्रमितुं शक्नुवन्ति।",
            'pa': "3.5 ਟਨ ਤੋਂ ਵੱਧ ਵਾਹਨਾਂ ਨੂੰ ਹੋਰ ਵਾਹਨਾਂ ਨੂੰ ਓਵਰਟੇਕ ਕਰਨ ਦੀ ਇਜਾਜ਼ਤ ਨਹੀਂ ਹੈ। ਹਲਕੇ ਵਾਹਨ ਅਜੇ ਵੀ ਓਵਰਟੇਕ ਕਰ ਸਕਦੇ ਹਨ।",
            'ta': "3.5 டன்களுக்கு மேற்பட்ட வாகனங்களால் பிற வாகனங்களை முந்திச் செல்ல அனுமதி இல்லை. இலகுவான வாகனங்கள் இன்னும் முந்திச் செல்லலாம்."
        }
    },
    11: {
        "name": "Right-of-way at intersection", 
        "guidance": {
            'en': "You have the right of way at the next intersection. Proceed with caution.",
            'es': "Tiene el derecho de paso en el próximo cruce. Proceda con precaución.",
            'fr': "Vous avez la priorité au prochain carrefour. Procédez avec prudence.",
            'hi': "अगले चौराहे पर आपको रास्ते का अधिकार है। सावधानी से आगे बढ़ें।",
            'ja': "次の交差点で優先権があります。注意して進んでください。",
            'kn': "ಮುಂದಿನ ಚೌಕಾಸ್ಥಳದಲ್ಲಿ ನಿಮಗೆ ಮುಂದುವರೆಯುವ ಹಕ್ಕಿದೆ. ಜಾಗರೂಕತೆಯಿಂದ ಮುಂದುವರಿಯಿರಿ.",
            'sa': "अग्रे चतुष्पथे भवतः गमनाधिकारः अस्ति। सावधानतया अग्रे गच्छतु।",
            'pa': "ਅਗਲੇ ਚੌਰਾਹੇ 'ਤੇ ਤੁਹਾਡੇ ਕੋਲ ਰਸਤੇ ਦਾ ਹੱਕ ਹੈ। ਸਾਵਧਾਨੀ ਨਾਲ ਅੱਗੇ ਵਧੋ।",
            'ta': "அடுத்த சந்திப்பில் உங்களுக்கு முன்னுரிமை உள்ளது. எச்சரிக்கையுடன் தொடரவும்."
        }
    },
    12: {
        "name": "Priority road", 
        "guidance": {
            'en': "You are on a priority road. Vehicles from side roads must yield to you.",
            'es': "Está en una carretera prioritaria. Los vehículos de las carreteras laterales deben cederle el paso.",
            'fr': "Vous êtes sur une route prioritaire. Les véhicules des routes secondaires doivent vous céder le passage.",
            'hi': "आप प्राथमिकता वाली सड़क पर हैं। साइड रोड से वाहनों को आपको रास्ता देना चाहिए।",
            'kn': "ನೀವು ಆದ್ಯತೆ ರಸ್ತೆಯಲ್ಲಿ ಇದ್ದೀರಿ. ಬದಿ ರಸ್ತೆಗಳಿಂದ ಬರುವ ವಾಹನಗಳು ನಿಮಗೆ ದಾರಿ ಕೊಡಬೇಕು.",
            'ja': "優先道路を走行中です。側道からの車両はあなたに道を譲らなければなりません。",
            'sa': "भवान् प्राथमिकमार्गे अस्ति। पार्श्वमार्गेभ्यः वाहनानि भवते मार्गं दातव्यम्।",
            'pa': "ਤੁਸੀਂ ਇੱਕ ਪ੍ਰਾਥਮਿਕਤਾ ਵਾਲੀ ਸੜਕ 'ਤੇ ਹੋ। ਸਾਈਡ ਰੋਡਾਂ ਤੋਂ ਗੱਡੀਆਂ ਨੂੰ ਤੁਹਾਨੂੰ ਰਸਤਾ ਦੇਣਾ ਚਾਹੀਦਾ ਹੈ।",
            'ta': "நீங்கள் முன்னுரிமை சாலையில் இருக்கிறீர்கள். பக்க சாலைகளிலிருந்து வாகனங்கள் உங்களுக்கு வழிவிட வேண்டும்."
        }
    },
    13: {
        "name": "Yield", 
        "guidance": {
            'en': "Yield to vehicles on the main road. Prepare to stop if necessary.",
            'es': "Ceda el paso a los vehículos en la carretera principal. Prepárese para detenerse si es necesario.",
            'fr': "Cédez le passage aux véhiculos sur la route principale. Préparez-vous à vous arrêter si nécessaire.",
            'hi': "मुख्य सड़क पर वाहनों को रास्ता दें। यदि आवश्यक हो तो रुकने के लिए तैयार रहें।",
            'ja': "幹線道路の車両に道を譲ってください。必要に応じて停車する準備をしてください。",
            'kn': "ಮುಖ್ಯ ರಸ್ತೆಯಲ್ಲಿರುವ ವಾಹನಗಳಿಗೆ ದಾರಿ ಕೊಡಿ. ಅಗತ್ಯವಿದ್ದಲ್ಲಿ ನಿಲ್ಲಲು ಸಿದ್ಧರಾಗಿರಿ.",
            'sa': "मुख्यमार्गे वाहनेभ्यः मार्गं ददातु। आवश्यकतायां स्थगनाय सज्जः भवतु।",
            'pa': "ਮੁੱਖ ਸੜਕ 'ਤੇ ਗੱਡੀਆਂ ਨੂੰ ਰਸਤਾ ਦਿਓ। ਜੇ ਲੋੜ ਪਵੇ ਤਾਂ ਰੁਕਣ ਲਈ ਤਿਆਰ ਰਹੋ।",
            'ta': "முக்கிய சாலையில் உள்ள வாகனங்களுக்கு வழிவிடவும். தேவைப்பட்டால் நிறுத்தத் தயாராக இருங்கள்."
        }
    },
    14: {
        "name": "Stop", 
        "guidance": {
            'en': "Come to a complete stop. Check all directions before proceeding.",
            'es': "Deténgase por completo. Verifique todas las direcciones antes de continuar.",
            'fr': "Arrêtez-vous complètement. Vérifiez toutes les directions avant de continuer.",
            'hi': "पूरी तरह से रुक जाएं। आगे बढ़ने से पहले सभी दिशाओं की जाँच करें।",
            'ja': "完全に停止してください。進行する前にすべての方向を確認してください。",
            'kn': "ಸಂಪೂರ್ಣವಾಗಿ ನಿಲ್ಲಿಸಿ. ಮುಂದುವರಿಯುವ ಮೊದಲು ಎಲ್ಲಾ ದಿಕ್ಕುಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
            'sa': "पूर्णतया स्थगयतु। अग्रे गमनात् पूर्वं सर्वाः दिशाः परीक्ष्यन्ताम्।",
            'pa': "ਪੂਰੀ ਤਰ੍ਹਾਂ ਰੁਕ ਜਾਓ। ਅੱਗੇ ਵਧਣ ਤੋਂ ਪਹਿਲਾਂ ਸਾਰੀਆਂ ਦਿਸ਼ਾਵਾਂ ਦੀ ਜਾਂਚ ਕਰੋ।",
            'ta': "முழுமையாக நிறுத்தவும். தொடர்வதற்கு முன் அனைத்து திசைகளையும் சரிபார்க்கவும்."
        }
    },
    15: {
        "name": "No vehicles", 
        "guidance": {
            'en': "No vehicles allowed beyond this point. This includes all motor vehicles.",
            'es': "No se permiten vehículos más allá de este punto. Esto incluye todos los vehículos motorizados.",
            'fr': "Aucun véhicule n'est autorisé au-delà de ce point. Cela inclut tous les véhicules à moteur.",
            'hi': "इस बिंदु से आगे कोई वाहन की अनुमति नहीं है। इसमें सभी मोटर वाहन शामिल हैं।",
            'kn': "ಈ ಹಂತದ ನಂತರ ಯಾವುದೇ ವಾಹನಗಳಿಗೆ ಅನುಮತಿ ಇಲ್ಲ. ಇದರಲ್ಲಿ ಎಲ್ಲಾ ಮೋಟಾರು ವಾಹನಗಳು ಸೇರಿವೆ.",
            'ja': "この先は車両通行禁止です。これにはすべての自動車が含まれます。",
            'sa': "अस्मात् बिन्दोः परं किमपि वाहनं अनुमतं नास्ति। इदं सर्वाणि मोटरवाहनानि अन्तर्भवति।",
            'pa': "ਇਸ ਬਿੰਦੂ ਤੋਂ ਬਾਅਦ ਕੋਈ ਵਾਹਨ ਮਨਜ਼ੂਰ ਨਹੀਂ ਹੈ। ਇਸ ਵਿੱਚ ਸਾਰੇ ਮੋਟਰ ਵਾਹਨ ਸ਼ਾਮਲ ਹਨ।",
            'ta': "இந்த இடத்திற்குப் பிறகு எந்த வாகனங்களும் அனுமதிக்கப்படவில்லை. இதில் அனைத்து மோட்டார் வாகனங்களும் அடங்கும்."
        }
    },
    16: {
        "name": "No vehicles over 3.5 tons", 
        "guidance": {
            'en': "Vehicles over 3.5 tons are prohibited. This includes trucks and heavy vehicles.",
            'es': "Se prohíben los vehículos de más de 3,5 toneladas. Esto incluye camiones y vehículos pesados.",
            'fr': "Les véhicules de plus de 3,5 tonnes sont interdits. Cela comprend les camions et les véhicules lourds.",
            'hi': "3.5 टन से अधिक के वाहन प्रतिबंधित हैं। इसमें ट्रक और भारी वाहन शामिल हैं।",
            'kn': "3.5 ಟನ್ಗಳಿಗಿಂತ ಹೆಚ್ಚು ವಾಹನಗಳು ನಿಷೇಧಿಸಲಾಗಿದೆ. ಇದರಲ್ಲಿ ಲಾರಿಗಳು ಮತ್ತು ಭಾರೀ ವಾಹನಗಳು ಸೇರಿವೆ.",
            'ja': "3.5トンを超える車両は禁止されています。これにはトラックや重量車両が含まれます。",
            'sa': "३.५ टनोपरिवाहनानि निषिद्धानि। इदं लॉर्यः भारीवाहनानि च अन्तर्भवति।",
            'pa': "3.5 ਟਨ ਤੋਂ ਵੱਧ ਵਾਹਨਾਂ 'ਤੇ ਪਾਬੰਦੀ ਹੈ। ਇਸ ਵਿੱਚ ਟਰੱਕ ਅਤੇ ਭਾਰੀ ਵਾਹਨ ਸ਼ਾਮਲ ਹਨ।",
            'ta': "3.5 டன்களுக்கு மேற்பட்ட வாகனங்கள் தடைசெய்யப்பட்டுள்ளன. இதில் லாரிகள் மற்றும் கனரக வாகனங்கள் அடங்கும்."
        }
    },
    17: {
        "name": "No entry", 
        "guidance": {
            'en': "Do not enter this road. This is a one-way street or restricted area.",
            'es': "No entre en esta carretera. Esta es una calle de un solo sentido o área restringida.",
            'fr': "N'entrez pas dans cette route. Il s'agit d'une rue à sens unique ou d'une zone restreinte.",
            'hi': "इस सड़क पर प्रवेश न करें। यह एकतरफा सड़क या प्रतिबंधित क्षेत्र है।",
            'ja': "この道路に入らないでください。これは一方通行または立入禁止区域です。",
            'kn': "ಈ ರಸ್ತೆಗೆ ಪ್ರವೇಶಿಸಬೇಡಿ. ಇದು ಒಂದೇ ದಿಕ್ಕಿನ ರಸ್ತೆ ಅಥವಾ ನಿರ್ಬಂಧಿತ ಪ್ರದೇಶವಾಗಿದೆ.",
            'sa': "अस्मिन् मार्गे प्रवेशं मा कुरुत। एषा एकदिशामार्गः प्रतिबन्धितक्षेत्रं वा अस्ति।",
            'pa': "ਇਸ ਸੜਕ 'ਤੇ ਦਾਖਲ ਨਾ ਹੋਵੋ। ਇਹ ਇੱਕ ਇਕੱਲੇ ਰਸਤੇ ਵਾਲੀ ਗਲੀ ਜਾਂ ਪਾਬੰਦੀਸ਼ੁਦਾ ਇਲਾਕਾ ਹੈ।",
            'ta': "இந்த சாலையில் நுழையாதீர்கள். இது ஒரே திசை சாலை அல்லது கட்டுப்படுத்தப்பட்ட பகுதி."
        }
    },
    18: {
        "name": "General caution", 
        "guidance": {
            'en': "General warning ahead. Be prepared for unexpected hazards.",
            'es': "Advertencia general adelante. Prepárese para peligros inesperados.",
            'fr': "Avertissement général devant. Soyez préparé à des dangers inattendus.",
            'hi': "सामान्य चेतावनी आगे। अप्रत्याशित खतरों के लिए तैयार रहें।",
            'kn': "ಮುಂದೆ ಸಾಮಾನ್ಯ ಎಚ್ಚರಿಕೆ. ಅನಿರೀಕ್ಷಿತ ಅಪಾಯಗಳಿಗೆ ಸಿದ್ಧರಾಗಿರಿ.",
            'ja': "前方に一般的な警告。予期しない危険に備えてください。",
            'sa': "सामान्यसूचना अग्रे। अनपेक्षितसंकटानां कृते सज्जः भवतु।",
            'pa': "ਅੱਗੇ ਸਾਧਾਰਨ ਚੇਤਾਵਨੀ। ਅਣਜਾਣ ਖਤਰਿਆਂ ਲਈ ਤਿਆਰ ਰਹੋ।",
            'ta': "முன்னால் பொது எச்சரிக்கை. எதிர்பாராத ஆபத்துகளுக்குத் தயாராக இருங்கள்."
        }
    },
    19: {
        "name": "Dangerous curve left", 
        "guidance": {
            'en': "Sharp curve to the left ahead. Reduce speed and stay in your lane.",
            'es': "Curva cerrada a la izquierda adelante. Reduzca la velocidad y manténgase en su carril.",
            'fr': "Virage dangereux à gauche devant. Réduisez votre vitesse et restez dans votre voie.",
            'hi': "बाईं ओर तेज मोड़ आगे। गति कम करें और अपनी लेन में रहें।",
            'kn': "ಮುಂದೆ ಎಡಕ್ಕೆ ತೀಕ್ಷ್ಣವಾದ ವಕ್ರರೇಖೆ. ವೇಗ ಕಡಿಮೆ ಮಾಡಿ ಮತ್ತು ನಿಮ್ಮ ಲೇನ್‌ನಲ್ಲಿ ಉಳಿಯಿರಿ.",
            'ja': "前方左カーブ注意。速度を落として車線を維持してください。",
            'sa': "अग्रे वामे तीव्रवक्रता। वेगं कमीकुरुत स्वीयपथे च तिष्ठतु।",
            'pa': "ਅੱਗੇ ਖੱਬੇ ਪਾਸੇ ਤੇਜ਼ ਮੋੜ। ਰਫ਼ਤਾਰ ਘਟਾਓ ਅਤੇ ਆਪਣੀ ਲੇਨ ਵਿੱਚ ਰਹੋ।",
            'ta': "முன்னால் இடதுபுறம் கூர்மையான வளைவு. வேகத்தைக் குறைத்து உங்கள் வழித்தடத்தில் இருங்கள்."
        }
    },
    20: {
        "name": "Dangerous curve right", 
        "guidance": {
            'en': "Sharp curve to the right ahead. Reduce speed and maintain control.",
            'es': "Curva cerrada a la derecha adelante. Reduzca la velocidad y mantenga el control.",
            'fr': "Virage dangereux à droite devant. Réduisez votre vitesse et maintenez le contrôle.",
            'hi': "दाईं ओर तेज मोड़ आगे। गति कम करें और नियंत्रण बनाए रखें।",
            'kn': "ಮುಂದೆ ಬಲಕ್ಕೆ ತೀಕ್ಷ್ಣವಾದ ವಕ್ರರೇಖೆ. ವೇಗ ಕಡಿಮೆ ಮಾಡಿ ಮತ್ತು ನಿಯಂತ್ರಣವನ್ನು维持 ಮಾಡಿಕೊಳ್ಳಿ.",
            'ja': "前方右カーブ注意。速度を落として制御を維持してください。",
            'sa': "अग्रे दक्षिणे तीव्रवक्रता। वेगं कमीकुरुत नियन्त्रणं च रक्षतु।",
            'pa': "ਅੱਗੇ ਸੱਜੇ ਪਾਸੇ ਤੇਜ਼ ਮੋੜ। ਰਫ਼ਤਾਰ ਘਟਾਓ ਅਤੇ ਕੰਟਰੋਲ ਬਣਾਈ ਰੱਖੋ।",
            'ta': "முன்னால் வலதுபுறம் கூர்மையான வளைவு. வேகத்தைக் குறைத்து கட்டுப்பாட்டைப் பேணுங்கள்."
        }
    },
    21: {
        "name": "Double curve", 
        "guidance": {
            'en': "Series of curves ahead, first to left then right or vice versa. Reduce speed significantly.",
            'es': "Series de curvas adelante, primero a la izquierda luego a la derecha o viceversa. Reduzca la velocidad significativamente.",
            'fr': "Série de virages devant, d'abord à gauche puis à droite ou vice versa. Réduisez considérablement votre vitesse.",
            'hi': "आगे मोड़ों की श्रृंखला, पहले बाएं फिर दाएं या इसके विपरीत। गति में काफी कमी करें।",
            'kn': "ಮುಂದೆ ವಕ್ರರೇಖೆಗಳ ಸರಣಿ, ಮೊದಲು ಎಡಕ್ಕೆ ನಂತರ ಬಲಕ್ಕೆ ಅಥವಾ ತದ್ವಿರುದ್ಧ. ವೇಗವನ್ನು ಗಣನೀಯವಾಗಿ ಕಡಿಮೆ ಮಾಡಿ.",
            'ja': "前方連続カーブ、左その後右またはその逆。速度を大幅に落としてください。",
            'sa': "अग्रे वक्राणां शृङ्खला, प्रथमं वामे ततः दक्षिणे वा विपरीतम्। वेगं महत्त्वपूर्णरूपेण कमीकुरुत।",
            'pa': "ਅੱਗੇ ਮੋੜਾਂ ਦੀ ਲੜੀ, ਪਹਿਲਾਂ ਖੱਬੇ ਫਿਰ ਸੱਜੇ ਜਾਂ ਇਸਦੇ ਉਲਟ। ਰਫ਼ਤਾਰ ਨੂੰ ਕਾਫ਼ੀ ਘਟਾਓ।",
            'ta': "முன்னால் தொடர் வளைவுகள், முதலில் இடது பின்னர் வலது அல்லது நேர்மாறாக. வேகத்தை கணிசமாகக் குறைக்கவும்."
        }
    },
    22: {
        "name": "Bumpy road", 
        "guidance": {
            'en': "Uneven or bumpy road surface ahead. Reduce speed and maintain firm grip on steering.",
            'es': "Superficie de carretera irregular o con baches adelante. Reduzca la velocidad y mantenga un firme control del volante.",
            'fr': "Surface de route irrégulière ou bosselée devant. Réduisez votre vitesse et maintenez une prise ferme sur le volant.",
            'hi': "आगे असमान या ऊबड़-खाबड़ सड़क की सतह। गति कम करें और स्टीयरिंग पर मजबूत पकड़ बनाए रखें।",
            'ja': "前方でんぼったまたは凸凹した路面。速度を落としてハンドルをしっかり握ってください。",
            'kn': "ಮುಂದೆ ಅಸಮ ರಸ್ತೆ ಮೇಲ್ಮೈ. ವೇಗ ಕಡಿಮೆ ಮಾಡಿ ಮತ್ತು ಸ್ಟೀರಿಂಗ್‌ನಲ್ಲಿ ದೃಢವಾದ ಹಿಡಿತವನ್ನು维持 ಮಾಡಿಕೊಳ್ಳಿ.",
            'sa': "अग्रे असमाना उबड़खाबड़ा वा मार्गतलम्। वेगं कमीकुरुत स्टीयरिंगे दृढपकड़ं च रक्षतु।",
            'pa': "ਅੱਗੇ ਅਸਮਾਨ ਜਾਂ ਉਬੜ-ਖਾਬੜ ਸੜਕ ਸਤਹ। ਰਫ਼ਤਾਰ ਘਟਾਓ ਅਤੇ ਸਟੀਅਰਿੰਗ 'ਤੇ ਮਜ਼ਬੂਤ ਪਕੜ ਬਣਾਈ ਰੱਖੋ।",
            'ta': "முன்னால் சீரற்ற அல்லது குழிவான சாலை மேற்பரப்பு. வேகத்தைக் குறைத்து ஸ்டீயரிங்கில் உறுதியான பிடியை பேணுங்கள்."
        }
    },
    23: {
        "name": "Slippery road", 
        "guidance": {
            'en': "Road may be slippery due to rain, ice, or other conditions. Reduce speed and avoid sudden maneuvers.",
            'es': "La carretera puede estar resbaladiza debido a la lluvia, el hielo u otras condiciones. Reduzca la velocidad y evite maniobras bruscas.",
            'fr': "La route peut être glissante à cause de la pluie, de la glace ou d'autres conditions. Réduisez votre vitesse et évitez les manœuvres brusques.",
            'hi': "बारिश, बर्फ या अन्य स्थितियों के कारण सड़क फिसलन भरी हो सकती है। गति कम करें और अचानक पैंतरेबाज़ी से बचें।",
            'ja': "雨、氷、その他の状態により道路が滑りやすくなっている可能性があります。速度を落として急な操作を避けてください。",
            'kn': "ಮಳೆ, ಬರ್ಫ ಅಥವಾ ಇತರ ಪರಿಸ್ಥಿತಿಗಳಿಂದಾಗಿ ರಸ್ತೆ ಜಾರುವಂತಿರಬಹುದು. ವೇಗ ಕಡಿಮೆ ಮಾಡಿ ಮತ್ತು ಹಠಾತ್ ಕಾರ್ಯಾಚರಣೆಗಳನ್ನು ತಪ್ಪಿಸಿ.",
            'sa': "वृष्टेः हिमस्य अन्यपरिस्थितीनां वा कारणेन मार्गः सर्पिष्टः भवेत्। वेगं कमीकुरुत आकस्मिकक्रियाः च परिहरतु।",
            'pa': "ਬਾਰਸ਼, ਬਰਫ਼ ਜਾਂ ਹੋਰ ਹਾਲਤਾਂ ਕਾਰਨ ਸੜਕ ਫਿਸਲਣ ਯੋਗ ਹੋ ਸਕਦੀ ਹੈ। ਰਫ਼ਤਾਰ ਘਟਾਓ ਅਤੇ ਅਚਾਨਕ ਪੈਂਤੜੇਬਾਜ਼ੀ ਤੋਂ ਬਚੋ।",
            'ta': "மழை, பனி அல்லது பிற நிலைமைகள் காரணமாக சாலை வழுக்கலாக இருக்கலாம். வேகத்தைக் குறைத்து திடீர் சமன்செயல்களைத் தவிர்க்கவும்."
        }
    },
    24: {
        "name": "Road narrows on the right", 
        "guidance": {
            'en': "Road narrows on the right side. Move left if safe to do so.",
            'kn': "ರಸ್ತೆ ಬಲಭಾಗದಲ್ಲಿ ಸಂಕುಚಿತಗೊಳ್ಳುತ್ತದೆ. ಸುರಕ್ಷಿತವಾಗಿದ್ದರೆ ಎಡಕ್ಕೆ ಸರಿಸಿ.",
            'es': "La carretera se estrecha en el lado derecho. Muévase a la izquierda si es seguro hacerlo.",
            'fr': "La route se rétrécit du côté droit. Déplacez-vous vers la gauche si c'est sans danger.",
            'hi': "दाईं ओर सड़क संकरी होती है। यदि सुरक्षित हो तो बाईं ओर बढ़ें।",
            'ja': "右側で道路が狭くなります。安全であれば左に移動してください。",
            'sa': "दक्षिणपार्श्वे मार्गः संकीर्णः भवति। सुरक्षितं चेत् वामे सरतु।",
            'pa': "ਸੜਕ ਸੱਜੇ ਪਾਸੇ ਸੰਕੀਰਨ ਹੋ ਜਾਂਦੀ ਹੈ। ਜੇ ਸੁਰੱਖਿਅਤ ਹੋਵੇ ਤਾਂ ਖੱਬੇ ਪਾਸੇ ਚਲੇ ਜਾਓ।",
            'ta': "சாலை வலது பக்கத்தில் குறுகுகிறது. பாதுகாப்பாக இருந்தால் இடதுபுறம் நகரவும்."
        }
    },
    25: {
        "name": "Road work", 
        "guidance": {
            'en': "Road construction ahead. Reduce speed, follow directions, and watch for workers.",
            'kn': "ಮುಂದೆ ರಸ್ತೆ ನಿರ್ಮಾಣ. ವೇಗ ಕಡಿಮೆ ಮಾಡಿ, ದಿಕ್ಕುಗಳನ್ನು ಅನುಸರಿಸಿ ಮತ್ತು ಕಾರ್ಮಿಕರಿಗಾಗಿ ನೋಡಿಕೊಳ್ಳಿ.",
            'es': "Construcción de carretera adelante. Reduzca la velocidad, siga las direcciones y esté atento a los trabajadores.",
            'fr': "Travaux routiers devant. Réduisez votre vitesse, suivez les directions et surveillez les travailleurs.",
            'hi': "आगे सड़क निर्माण। गति कम करें, निर्देशों का पालन करें और श्रमिकों पर नजर रखें।",
            'ja': "前方で道路工事中。速度を落とし、指示に従い、作業員に注意してください。",
            'sa': "अग्रे मार्गनिर्माणम्। वेगं कमीकुरुत निर्देशान् अनुसरतु कर्मचारिषु च सावधानः भवतु।",
            'pa': "ਅੱਗੇ ਸੜਕ ਨਿਰਮਾਣ। ਰਫ਼ਤਾਰ ਘਟਾਓ, ਨਿਰਦੇਸ਼ਾਂ ਦੀ ਪਾਲਣਾ ਕਰੋ ਅਤੇ ਕਰਮਚਾਰੀਆਂ ਲਈ ਧਿਆਨ ਰੱਖੋ।",
            'ta': "முன்னால் சாலை பணிகள். வேகத்தைக் குறைத்து, வழிகாட்டுதல்களைப் பின்பற்றி, தொழிலாளர்களைக் கவனிக்கவும்."
        }
    },
    26: {
        "name": "Traffic signals", 
        "guidance": {
            'en': "Traffic lights ahead. Be prepared to stop if light is red or changing.",
            'es': "Semáforos adelante. Prepárese para detenerse si la luz está en rojo o cambiando.",
            'fr': "Feux de circulation devant. Soyez prêt à vous arrêter si le feu est rouge ou change.",
            'hi': "आगे ट्रैफिक लाइट। यदि लाल बत्ती है या बदल रही है तो रुकने के लिए तैयार रहें।",
            'ja': "前方に信号機。信号が赤または変わりそうな場合は停車する準備をしてください。",
            'kn': "ಮುಂದೆ ಟ್ರಾಫಿಕ್ ಲೈಟ್ಗಳು. ಬೆಳಕು ಕೆಂಪಗಿದ್ದರೆ ಅಥವಾ ಬದಲಾಗುತ್ತಿದ್ದರೆ ನಿಲ್ಲಲು ಸಿದ್ಧರಾಗಿರಿ.",
            'sa': "अग्रे यातायातप्रकाशाः। प्रकाशः रक्तः परिवर्तनशीलः वा चेत् स्थगनाय सज्जः भवतु।",
            'pa': "ਅੱਗੇ ਟ੍ਰੈਫਿਕ ਲਾਈਟਾਂ। ਜੇ ਲਾਈਟ ਲਾਲ ਹੈ ਜਾਂ ਬਦਲ ਰਹੀ ਹੈ ਤਾਂ ਰੁਕਣ ਲਈ ਤਿਆਰ ਰਹੋ।",
            'ta': "முன்னால் போக்குவரத்து சமிக்ஞைகள். ஒளி சிவப்பு நிறமாக இருந்தால் அல்லது மாறுவதாக இருந்தால் நிறுத்தத் தயாராக இருங்கள்."
        }
    },
    27: {
        "name": "Pedestrians", 
        "guidance": {
            'en': "Pedestrian crossing area ahead. Reduce speed and be prepared to stop for people crossing.",
            'es': "Área de cruce de peatones adelante. Reduzca la velocidad y esté preparado para detenerse para las personas que cruzan.",
            'fr': "Zone de passage pour piétons devant. Réduisez votre vitesse et soyez prêt à vous arrêter pour les piétons qui traversent.",
            'hi': "आगे पैदल यात्री क्रॉसिंग क्षेत्र। गति कम करें और पार करने वाले लोगों के लिए रुकने के लिए तैयार रहें।",
            'ja': "前方横断歩道エリア。速度を落として横断する人のために停車する準備をしてください。",
            'kn': "ಮುಂದೆ ಪಾದಚಾರಿ ದಾಟುವ ಪ್ರದೇಶ. ವೇಗ ಕಡಿಮೆ ಮಾಡಿ ಮತ್ತು ದಾಟುವ ಜನರಿಗಾಗಿ ನಿಲ್ಲಲು ಸಿದ್ಧರಾಗಿರಿ.",
            'sa': "अग्रे पादचारिसङ्क्रमणक्षेत्रम्। वेगं कमीकुरुत सङ्क्रममाणजनानां कृते स्थगनाय सज्जः भवतु।",
            'pa': "ਅੱਗੇ ਪੈਦਲ ਚਲਣ ਵਾਲਿਆਂ ਦਾ ਪਾਰ ਕਰਨ ਵਾਲਾ ਖੇਤਰ। ਰਫ਼ਤਾਰ ਘਟਾਓ ਅਤੇ ਪਾਰ ਕਰਨ ਵਾਲੇ ਲੋਕਾਂ ਲਈ ਰੁਕਣ ਲਈ ਤਿਆਰ ਰਹੋ।",
            'ta': "முன்னால் பாதசாரி கடக்கும் பகுதி. வேகத்தைக் குறைத்து, கடக்கும் மக்களுக்காக நிறுத்தத் தயாராக இருங்கள்."
        }
    },
    28: {
        "name": "Children crossing", 
        "guidance": {
            'en': "School zone or area with children. Reduce speed significantly and be extra cautious.",
            'es': "Zona escolar o área con niños. Reduzca la velocidad significativamente y extreme las precauciones.",
            'fr': "Zone scolaire ou zone avec des enfants. Réduisez considérablement votre vitesse et soyez très prudent.",
            'hi': "स्कूल जोन या बच्चों वाला क्षेत्र। गति में काफी कमी करें और अतिरिक्त सावधानी बरतें।",
            'kn': "ಶಾಲಾ ವಲಯ ಅಥವಾ ಮಕ್ಕಳಿರುವ ಪ್ರದೇಶ. ವೇಗವನ್ನು ಗಣನೀಯವಾಗಿ ಕಡಿಮೆ ಮಾಡಿ ಮತ್ತು ಹೆಚ್ಚುವರಿ ಜಾಗರೂಕರಾಗಿರಿ.",
            'ja': "スクールゾーンまたは子供がいるエリア。速度を大幅に落として特に注意してください。",
            'sa': "विद्यालयक्षेत्रं बालकयुक्तक्षेत्रं वा। वेगं महत्त्वपूर्णरूपेण कमीकुरुत अतिरिक्तसावधानः च भवतु।",
            'pa': "ਸਕੂਲ ਜ਼ੋਨ ਜਾਂ ਬੱਚਿਆਂ ਵਾਲਾ ਇਲਾਕਾ। ਰਫ਼ਤਾਰ ਨੂੰ ਕਾਫ਼ੀ ਘਟਾਓ ਅਤੇ ਵਿਸ਼ੇਸ਼ ਸਾਵਧਾਨ ਰਹੋ।",
            'ta': "பள்ளி மண்டலம் அல்லது குழந்தைகள் உள்ள பகுதி. வேகத்தை கணிசமாகக் குறைத்து கூடுதல் எச்சரிக்கையாக இருங்கள்."
        }
    },
    29: {
        "name": "Bicycles crossing", 
        "guidance": {
            'en': "Bicycle crossing area. Watch for cyclists and give them space.",
            'es': "Área de cruce de bicicletas. Esté atento a los ciclistas y déles espacio.",
            'fr': "Zone de passage pour vélos. Surveillez les cyclistes et donnez-leur de l'espace.",
            'hi': "साइकिल क्रॉसिंग क्षेत्र। साइकिल चालकों पर नजर रखें और उन्हें जगह दें।",
            'kn': "ಸೈಕಲ್ ದಾಟುವ ಪ್ರದೇಶ. ಸೈಕಲ್ ಸವಾರರಿಗಾಗಿ ನೋಡಿಕೊಳ್ಳಿ ಮತ್ತು ಅವರಿಗೆ ಜಾಗವನ್ನು ನೀಡಿ.",
            'ja': "自転車横断エリア。自転車に注意し、スペースを確保してください。",
            'sa': "साइकिलसङ्क्रमणक्षेत्रम्। साइकिलचालकान् प्रति सजगः भवतु तेभ्यः स्थानं च ददातु।",
            'pa': "ਸਾਈਕਲ ਪਾਰ ਕਰਨ ਵਾਲਾ ਖੇਤਰ। ਸਾਈਕਲ ਸਵਾਰਾਂ ਲਈ ਧਿਆਨ ਰੱਖੋ ਅਤੇ ਉਨ੍ਹਾਂ ਨੂੰ ਜਗ੍ਹਾ ਦਿਓ।",
            'ta': "மிதிவண்டி கடக்கும் பகுதி. மிதிவண்டி ஓட்டிகளைக் கவனித்து, அவர்களுக்கு இடம் கொடுக்கவும்."
        }
    },
    30: {
        "name": "Beware of ice/snow", 
        "guidance": {
            'en': "Icy or snowy conditions possible. Reduce speed, increase following distance, and avoid sudden moves.",
            'es': "Posibles condiciones de hielo o nieve. Reduzca la velocidad, aumente la distancia de seguimiento y evite movimientos bruscos.",
            'fr': "Conditions glacées ou neigeuses possibles. Réduisez votre vitesse, augmentez la distance de suivi et évitez les mouvements brusques.",
            'hi': "बर्फीली या बर्फीली स्थितियाँ संभव हैं। गति कम करें, फॉलो करने की दूरी बढ़ाएं और अचानक चाल से बचें।",
            'ja': "氷や雪の状態の可能性。速度を落とし、車間距離を広げ、急な動きを避けてください。",
            'kn': "ಬರ್ಫ ಅಥವಾ ಹಿಮದ ಪರಿಸ್ಥಿತಿಗಳ ಸಾಧ್ಯತೆ. ವೇಗ ಕಡಿಮೆ ಮಾಡಿ, ಅನುಸರಣೆ ದೂರವನ್ನು ಹೆಚ್ಚಿಸಿ ಮತ್ತು ಹಠಾತ್ ಚಲನೆಗಳನ್ನು ತಪ್ಪಿಸಿ.",
            'sa': "हिमिलाः हिमयुक्ताः वा परिस्थितयः सम्भवनीयाः। वेगं कमीकुरुत अनुगमनदूरीं वर्धयतु आकस्मिकचलनानि च परिहरतु।",
            'pa': "ਬਰਫ਼ੀਲੀਆਂ ਜਾਂ ਬਰਫ਼ੀਲੀਆਂ ਹਾਲਤਾਂ ਸੰਭਵ ਹਨ। ਰਫ਼ਤਾਰ ਘਟਾਓ, ਪਿੱਛੇ ਚਲਣ ਦੀ ਦੂਰੀ ਵਧਾਓ ਅਤੇ ਅਚਾਨਕ ਚਾਲਾਂ ਤੋਂ ਬਚੋ।",
            'ta': "பனி அல்லது பனி நிலைமைகள் சாத்தியம். வேகத்தைக் குறைத்து, பின்தொடரும் தூரத்தை அதிகரித்து, திடீர் நகர்வுகளைத் தவிர்க்கவும்."
        }
    },
    31: {
        "name": "Wild animals crossing", 
        "guidance": {
            'en': "Area with wild animal crossings. Be especially cautious at dawn and dusk.",
            'es': "Área con cruces de animales salvajes. Extreme las precauciones al amanecer y al anochecer.",
            'fr': "Zone avec passages d'animaux sauvages. Soyez particulièrement prudent à l'aube et au crépuscule.",
            'hi': "जंगली जानवरों के क्रॉसिंग वाला क्षेत्र। सुबह और शाम के समय विशेष सावधानी बरतें।",
            'kn': "ಕಾಡು ಪ್ರಾಣಿಗಳ ದಾಟುವ ಪ್ರದೇಶ. ಮುಂಜಾನೆ ಮತ್ತು ಸಂಜೆ ವಿಶೇಷವಾಗಿ ಜಾಗರೂಕರಾಗಿರಿ.",
            'ja': "野生動物の横断エリア。夕暮れ時と夜明けには特に注意してください。",
            'sa': "वन्यपशुसङ्क्रमणक्षेत्रम्। प्रभाते सायंकाले च विशेषतया सावधानः भवतु।",
            'pa': "ਜੰਗਲੀ ਜਾਨਵਰਾਂ ਦੇ ਪਾਰ ਕਰਨ ਵਾਲਾ ਇਲਾਕਾ। ਸਵੇਰ ਅਤੇ ਸ਼ਾਮ ਨੂੰ ਖਾਸ ਸਾਵਧਾਨ ਰਹੋ।",
            'ta': "காட்டு விலங்குகள் கடக்கும் பகுதி. வைகறை மற்றும் மாலை நேரங்களில் க特别 எச்சரிக்கையாக இருங்கள்."
        }
    },
    32: {
        "name": "End of all speed and passing limits", 
        "guidance": {
            'en': "All previous speed and passing restrictions have ended. Still drive according to road conditions.",
            'es': "Todas las restricciones anteriores de velocidad y adelantamiento han terminado. Aún conduzca de acuerdo con las condiciones de la carretera.",
            'fr': "Toutes les restrictions précédentes de vitesse et de dépassement sont terminées. Conduisez toujours en fonction des conditions routières.",
            'hi': "पिछली सभी गति और ओवरटेकिंग प्रतिबंध समाप्त हो गए हैं। फिर भी सड़क की स्थिति के अनुसार गाड़ी चलाएं।",
            'ja': "これまでのすべての速度制限と追い越し制限は終了しました。それでも道路状況に応じて運転してください。",
            'kn': "ಎಲ್ಲಾ ಮುಂಚಿನ ವೇಗ ಮತ್ತು ಮುಂದೆ ಹೋಗುವ ನಿರ್ಬಂಧಗಳು ಮುಗಿದಿವೆ. ಇನ್ನೂ ರಸ್ತೆ ಪರಿಸ್ಥಿತಿಗಳಿಗೆ ಅನುಗುಣವಾಗಿ ಚಾಲನೆ ಮಾಡಿ.",
            'sa': "सर्वाः पूर्ववेगअतिक्रमणनिर्बन्धाः समाप्ताः। तथापि मार्गपरिस्थितीनुसारं चालयतु।",
            'pa': "ਪਿਛਲੀਆਂ ਸਾਰੀਆਂ ਗਤੀ ਅਤੇ ਓਵਰਟੇਕਿੰਗ ਪਾਬੰਦੀਆਂ ਖਤਮ ਹੋ ਗਈਆਂ ਹਨ। ਫਿਰ ਵੀ ਸੜਕ ਹਾਲਤਾਂ ਅਨੁਸਾਰ ਗੱਡੀ ਚਲਾਓ।",
            'ta': "முன்னைய அனைத்து வேக மற்றும் முந்திச் செல்லும் கட்டுப்பாடுகள் முடிவடைந்தன. இன்னும் சாலை நிலைமைகளுக்கு ஏறவே வாகனம் ஓட்டுங்கள்."
        }
    },
    33: {
        "name": "Turn right ahead", 
        "guidance": {
            'en': "Mandatory right turn ahead. Prepare to turn right at the intersection.",
            'es': "Giro a la derecha obligatorio adelante. Prepárese para girar a la derecha en el cruce.",
            'fr': "Virage à droite obligatoire devant. Préparez-vous à tourner à droite à l'intersection.",
            'hi': "आगे अनिवार्य दाएं मुड़ें। चौराहे पर दाएं मुड़ने के लिए तैयार हो जाएं।",
            'kn': "ಮುಂದೆ ಕಡ್ಡಾಯ ಬಲ ತಿರುಗುವಿಕೆ. ಚೌಕಾಸ್ಥಳದಲ್ಲಿ ಬಲಕ್ಕೆ ತಿರುಗಲು ಸಿದ್ಧರಾಗಿರಿ.",
            'ja': "前方右折義務。交差点で右折する準備をしてください。",
            'sa': "अग्रे अनिवार्यदक्षिणवर्तनम्। चतुष्पथे दक्षिणे वर्तनाय सज्जः भवतु।",
            'pa': "ਅੱਗੇ ਲਾਜ਼ਮੀ ਸੱਜੇ ਮੋੜ। ਚੌਰਾਹੇ 'ਤੇ ਸੱਜੇ ਮੁੜਨ ਲਈ ਤਿਆਰ ਹੋ ਜਾਓ।",
            'ta': "முன்னால் கட்டாய வலது திருப்பம். சந்திப்பில் வலதுபுறம் திரும்பத் தயாராக இருங்கள்."
        }
    },
    34: {
        "name": "Turn left ahead", 
        "guidance": {
            'en': "Mandatory left turn ahead. Prepare to turn left at the intersection.",
            'es': "Giro a la izquierda obligatorio adelante. Prepárese para girar a la izquierda en el cruce.",
            'fr': "Virage à gauche obligatoire devant. Préparez-vous à tourner à gauche à l'intersection.",
            'hi': "आगे अनिवार्य बाएं मुड़ें। चौराहे पर बाएं मुड़ने के लिए तैयार हो जाएं।",
            'kn': "ಮುಂದೆ ಕಡ್ಡಾಯ ಎಡ ತಿರುಗುವಿಕೆ. ಚೌಕಾಸ್ಥಳದಲ್ಲಿ ಎಡಕ್ಕೆ ತಿರುಗಲು ಸಿದ್ಧರಾಗಿರಿ.",
            'ja': "前方左折義務。交差点で左折する準備をしてください。",
            'sa': "अग्रे अनिवार्यवामवर्तनम्। चतुष्पथे वामे वर्तनाय सज्जः भवतु।",
            'pa': "ਅੱਗੇ ਲਾਜ਼ਮੀ ਖੱਬੇ ਮੋੜ। ਚੌਰਾਹੇ 'ਤੇ ਖੱਬੇ ਮੁੜਨ ਲਈ ਤਿਆਰ ਹੋ ਜਾਓ।",
            'ta': "முன்னால் கட்டாய இடது திருப்பம். சந்திப்பில் இடதுபுறம் திரும்பத் தயாராக இருங்கள்."
        }
    },
    35: {
        "name": "Ahead only", 
        "guidance": {
            'en': "You must continue straight ahead. No turns allowed at this intersection.",
            'es': "Debe continuar recto. No se permiten giros en este cruce.",
            'fr': "Vous devez continuer tout droit. Aucun virage n'est autorisé à cette intersection.",
            'hi': "आपको सीधे आगे बढ़ना होगा। इस चौराहे पर कोई मोड़ की अनुमति नहीं है।",
            'kn': "ನೀವು ನೇರವಾಗಿ ಮುಂದುವರಿಯಬೇಕು. ಈ ಚೌಕಾಸ್ಥಳದಲ್ಲಿ ಯಾವುದೇ ತಿರುಗುವಿಕೆಗಳಿಗೆ ಅನುಮತಿ ಇಲ್ಲ.",
            'ja': "直進のみ。この交差点では曲がり禁止。",
            'sa': "भवान् सीधे अग्रे एव गच्छेत्। अस्य चतुष्पथे कापि वर्तनानि अनुमतानि न सन्ति।",
            'pa': "ਤੁਹਾਨੂੰ ਸਿੱਧੇ ਅੱਗੇ ਹੀ ਜਾਰੀ ਰੱਖਣਾ ਚਾਹੀਦਾ ਹੈ। ਇਸ ਚੌਰਾਹੇ 'ਤੇ ਕੋਈ ਮੋੜ ਮਨਜ਼ੂਰ ਨਹੀਂ ਹੈ।",
            'ta': "நீங்கள் நேராக முன்னே செல்ல வேண்டும். இந்த சந்திப்பில் எந்தத் திருப்பங்களும் அனுமதிக்கப்படவில்லை."
        }
    },
    36: {
        "name": "Go straight or right", 
        "guidance": {
            'en': "You may go straight or turn right. Left turns are not permitted.",
            'es': "Puede ir recto o girar a la derecha. No se permiten giros a la izquierda.",
            'fr': "Vous pouvez aller tout droit ou tourner à droite. Les virages à gauche ne sont pas autorisés.",
            'hi': "आप सीधे जा सकते हैं या दाएं मुड़ सकते हैं। बाएं मोड़ की अनुमति नहीं है।",
            'kn': "ನೀವು ನೇರವಾಗಿ ಹೋಗಬಹುದು ಅಥವಾ ಬಲಕ್ಕೆ ತಿರುಗಬಹುದು. ಎಡ ತಿರುಗುವಿಕೆಗಳಿಗೆ ಅನುಮತಿ ಇಲ್ಲ.",
            'ja': "直進または右折可能。左折は禁止されています。",
            'sa': "भवान् सीधे गच्छेत् दक्षिणे वा वर्तेत। वामवर्तनानि अनुमतानि न सन्ति।",
            'pa': "ਤੁਸੀਂ ਸਿੱਧੇ ਜਾ ਸਕਦੇ ਹੋ ਜਾਂ ਸੱਜੇ ਮੁੜ ਸਕਦੇ ਹੋ। ਖੱਬੇ ਮੋੜ ਦੀ ਇਜਾਜ਼ਤ ਨਹੀਂ ਹੈ।",
            'ta': "நீங்கள் நேராகச் செல்லலாம் அல்லது வலதுபுறம் திரும்பலாம். இடதுபுறத் திருப்பங்கள் அனுமதிக்கப்படவில்லை."
        }
    },
    37: {
        "name": "Go straight or left", 
        "guidance": {
            'en': "You may go straight or turn left. Right turns are not permitted.",
            'es': "Puede ir recto o girar a la izquierda. No se permiten giros a la derecha.",
            'fr': "Vous pouvez aller tout droit ou tourner à gauche. Les virages à droite ne sont pas autorisés.",
            'hi': "आप सीधे जा सकते हैं या बाएं मुड़ सकते हैं। दाएं मोड़ की अनुमति नहीं है।",
            'kn': "ನೀವು ನೇರವಾಗಿ ಹೋಗಬಹುದು ಅಥವಾ ಎಡಕ್ಕೆ ತಿರುಗಬಹುದು. ಬಲ ತಿರುಗುವಿಕೆಗಳಿಗೆ ಅನುಮತಿ ಇಲ್ಲ.",
            'ja': "直進または左折可能。右折は禁止されています。",
            'sa': "भवान् सीधे गच्छेत् वामे वा वर्तेत। दक्षिणवर्तनानि अनुमतानि न सन्ति।",
            'pa': "ਤੁਸੀਂ ਸਿੱਧੇ ਜਾ ਸਕਦੇ ਹੋ ਜਾਂ ਖੱਬੇ ਮੁੜ ਸਕਦੇ ਹੋ। ਸੱਜੇ ਮੋੜ ਦੀ ਇਜਾਜ਼ਤ ਨਹੀਂ ਹੈ।",
            'ta': "நீங்கள் நேராகச் செல்லலாம் அல்லது இடதுபுறம் திரும்பலாம். வலதுபுறத் திருப்பங்கள் அனுமதிக்கப்படவில்லை."
        }
    },
    38: {
        "name": "Keep right", 
        "guidance": {
            'en': "Keep to the right side of the road. This may indicate a divider or obstacle ahead.",
            'es': "Manténgase en el lado derecho de la carretera. Esto puede indicar un divisor u obstáculo adelante.",
            'fr': "Restez sur le côté droit de la route. Cela peut indiquer un séparateur ou un obstacle devant.",
            'hi': "सड़क के दाईं ओर रहें। यह आगे एक विभाजक या बाधा का संकेत दे सकता है।",
            'ja': "道路の右側を維持してください。これは前方の仕切りや障害物を示している可能性があります。",
            'kn': "ರಸ್ತೆಯ ಬಲಭಾಗದಲ್ಲಿ ಉಳಿಯಿರಿ. ಇದು ಮುಂದೆ ವಿಭಜಕ ಅಥವಾ ಅಡಚಣೆಯನ್ನು ಸೂಚಿಸಬಹುದು.",
            'sa': "मार्गस्य दक्षिणपार्श्वे तिष्ठतु। इदं अग्रे विभाजकं बाधां वा सूचयेत्।",
            'pa': "ਸੜਕ ਦੇ ਸੱਜੇ ਪਾਸੇ ਰਹੋ। ਇਹ ਅੱਗੇ ਵੰਡਣ ਵਾਲਾ ਜਾਂ ਰੁਕਾਵਟ ਦਾ ਸੰਕੇਤ ਹੋ ਸਕਦਾ ਹੈ।",
            'ta': "சாலையின் வலது பக்கத்தில் இருங்கள். இது முன்னால் ஒரு பிரிப்பான் அல்லது தடையைக் குறிக்கலாம்."
        }
    },
    39: {
        "name": "Keep left", 
        "guidance": {
            'en': "Keep to the left side of the road. This may indicate a divider or obstacle ahead.",
            'es': "Manténgase en el lado izquierdo de la carretera. Esto puede indicar un divisor u obstáculo adelante.",
            'fr': "Restez sur le côté gauche de la route. Cela peut indiquer un séparateur ou un obstacle devant.",
            'hi': "सड़क के बाईं ओर रहें। यह आगे एक विभाजक या बाधा का संकेत दे सकता है।",
            'kn': "ರಸ್ತೆಯ ಎಡಭಾಗದಲ್ಲಿ ಉಳಿಯಿರಿ. ಇದು ಮುಂದೆ ವಿಭಜಕ ಅಥವಾ ಅಡಚಣೆಯನ್ನು ಸೂಚಿಸಬಹುದು.",
            'ja': "道路の左側を維持してください。これは前方の仕切りや障害物を示している可能性があります।",
            'sa': "मार्गस्य वामपार्श्वे तिष्ठतु। इदं अग्रे विभाजकं बाधां वा सूचयेत्।",
            'pa': "ਸੜਕ ਦੇ ਖੱਬੇ ਪਾਸੇ ਰਹੋ। ਇਹ ਅੱਗੇ ਵੰਡਣ ਵਾਲਾ ਜਾਂ ਰੁਕਾਵਟ ਦਾ ਸੰਕੇਤ ਹੋ ਸਕਦਾ ਹੈ।",
            'ta': "சாலையின் இடது பக்கத்தில் இருங்கள். இது முன்னால் ஒரு பிரிப்பான் அல்லது தடையைக் குறிக்கலாம்."
        }
    },
    40: {
        "name": "Roundabout mandatory", 
        "guidance": {
            'en': "Roundabout ahead. Yield to traffic already in the roundabout and proceed counter-clockwise.",
            'es': "Rotonda adelante. Ceda el paso al tráfico que ya está en la rotonda y proceda en sentido antihorario.",
            'fr': "Rond-point devant. Cédez le passage à la circulation déjà dans le rond-point et procédez dans le sens inverse des aiguilles d'une montre.",
            'hi': "आगे राउंडअबाउट। राउंडअबाउट में पहले से मौजूद ट्रैफिक को रास्ता दें और वामावर्त आगे बढ़ें।",
            'kn': "ಮುಂದೆ ರೌಂಡಾಬೌಟ್. ರೌಂಡಾಬೌಟ್‌ನಲ್ಲಿ ಈಗಾಗಲೇ ಇರುವ ಟ್ರಾಫಿಕ್‌ಗೆ ದಾರಿ ಕೊಡಿ ಮತ್ತು ಎಡಭಿಮುಖವಾಗಿ ಮುಂದುವರಿಯಿರಿ.",
            'ja': "前方ラウンドアバウト。ラウンドアバウト内の交通に道を譲り、反時計回りに進んでください。",
            'sa': "अग्रे चक्रमार्गः। चक्रमार्गे स्थितयातायाताय मार्गं ददातु वामावर्तं च अग्रे गच्छतु।",
            'pa': "ਅੱਗੇ ਰਾਊਂਡਅਬਾਊਟ। ਰਾਊਂਡਅਬਾਊਟ ਵਿੱਚ ਪਹਿਲਾਂ ਹੀ ਮੌਜੂਦ ਟ੍ਰੈਫਿਕ ਨੂੰ ਰਸਤਾ ਦਿਓ ਅਤੇ ਘੜੀ ਦੇ ਉਲਟ ਦਿਸ਼ਾ ਵਿੱਚ ਅੱਗੇ ਵਧੋ।",
            'ta': "முன்னால் சுற்றுவட்டப் பாதை. சுற்றுவட்டப் பாதையில் ஏற்கனவே உள்ள போக்குவரத்துக்கு வழிவிடவும், எதிர் கடிகார திசையில் தொடரவும்."
        }
    },
    41: {
        "name": "End of no passing", 
        "guidance": {
            'en': "No passing restrictions have ended. You may now pass other vehicles where safe and legal.",
            'es': "Las restricciones de no adelantar han terminado. Ahora puede adelantar a otros vehículos donde sea seguro y legal.",
            'fr': "Les restrictions de non-dépassement sont terminées. Vous pouvez maintenant dépasser d'autres véhicules là où c'est sûr et légal.",
            'hi': "नो पासिंग प्रतिबंध समाप्त हो गए हैं। अब आप सुरक्षित और कानूनी जगह पर अन्य वाहनों को पास कर सकते हैं।",
            'ja': "追い越し禁止制限は終了しました。安全で合法的な場所で他の車両を追い越せるようになりました。",
            'kn': "ಮುಂದೆ ಹೋಗುವ ನಿರ್ಬಂಧಗಳು ಮುಗಿದಿವೆ. ನೀವು ಈಗ ಸುರಕ್ಷಿತ ಮತ್ತು ಕಾನೂನುಬದ್ಧವಾದಲ್ಲಿ ಇತರ ವಾಹನಗಳನ್ನು ಮುಂದೆ ಹೋಗಬಹುದು.",
            'sa': "अतिक्रमणनिर्बन्धाः समाप्ताः। भवान् इदानीं सुरक्षिते कानूनीस्थले च अन्यवाहनानि अतिक्रमितुं शक्नोति।",
            'pa': "ਓਵਰਟੇਕਿੰਗ 'ਤੇ ਪਾਬੰਦੀਆਂ ਖਤਮ ਹੋ ਗਈਆਂ ਹਨ। ਹੁਣ ਤੁਸੀਂ ਸੁਰੱਖਿਅਤ ਅਤੇ ਕਾਨੂੰਨੀ ਜਗ੍ਹਾ 'ਤੇ ਹੋਰ ਵਾਹਨਾਂ ਨੂੰ ਓਵਰਟੇਕ ਕਰ ਸਕਦੇ ਹੋ।",
            'ta': "முந்திச் செல்லும் தடைகள் முடிவடைந்தன. இப்போது நீங்கள் பாதுகாப்பான மற்றும் சட்டபூர்வமான இடங்களில் பிற வாகனங்களை முந்திச் செல்லலாம்."
        }
    },
    42: {
        "name": "End of no passing for vehicles over 3.5 tons", 
        "guidance": {
            'en': "Passing restrictions for heavy vehicles have ended. Trucks may now pass where permitted.",
            'es': "Las restricciones de adelantamiento para vehículos pesados han terminado. Los camiones ahora pueden adelantar donde esté permitido.",
            'fr': "Les restrictions de dépassement pour les véhicules lourds ont pris fin. Les camions peuvent maintenant dépasser là où c'est autorisé.",
            'hi': "भारी वाहनों के लिए ओवरटेकिंग प्रतिबंध समाप्त हो गए हैं। ट्रक अब उन जगहों पर ओवरटेक कर सकते हैं जहाँ अनुमति है।",
            'ja': "重量車両の追い越し制限が終了しました。トラックは許可されている場所で追い越し可能です。",
            'kn': "ಭಾರೀ ವಾಹನಗಳಿಗೆ ಮುಂದೆ ಹೋಗುವ ನಿರ್ಬಂಧಗಳು ಮುಗಿದಿವೆ. ಲಾರಿಗಳು ಈಗ ಅನುಮತಿ ಇರುವಲ್ಲಿ ಮುಂದೆ ಹೋಗಬಹುದು.",
            'sa': "भारीवाहनानाम् अतिक्रमणनिर्बन्धाः समाप्ताः। लॉर्यः इदानीं अनुमतस्थले अतिक्रमितुं शक्नुवन्ति।",
            'pa': "ਭਾਰੀ ਵਾਹਨਾਂ ਲਈ ਓਵਰਟੇਕਿੰਗ ਪਾਬੰਦੀਆਂ ਖਤਮ ਹੋ ਗਈਆਂ ਹਨ। ਟਰੱਕ ਹੁਣ ਉਨ੍ਹਾਂ ਜਗ੍ਹਾਵਾਂ 'ਤੇ ਓਵਰਟੇਕ ਕਰ ਸਕਦੇ ਹਨ ਜਿੱਥੇ ਇਜਾਜ਼ਤ ਹੈ।",
            'ta': "கனரக வாகனங்களுக்கான முந்திச் செல்லும் தடைகள் முடிவடைந்தன. லாரிகள் இப்போது அனுமதிக்கப்பட்ட இடங்களில் முந்திச் செல்லலாம்."
        }
    }
}
def preprocess_image(image):
    """Preprocess the image for model prediction"""
    image = image.resize((30, 30))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_traffic_sign(image):
    """Predict traffic sign from image"""
    if model is None:
        return None, None, None, None
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    # Get top 3 predictions
    top3_indices = np.argsort(prediction[0])[-3:][::-1]
    top3_confidences = prediction[0][top3_indices]
    top3_classes = [classes[i] for i in top3_indices]
    
    top_predictions = []
    for cls, conf in zip(top3_classes, top3_confidences):
        top_predictions.append({
            "class": cls["name"], 
            "confidence": float(conf), 
            "guidance": cls["guidance"]["en"]  # Default to English for top predictions
        })
    
    return classes[predicted_class]["name"], float(confidence), top_predictions, classes[predicted_class]["guidance"]

def text_to_speech(text, lang_code='en'):
    """Convert text to speech and return base64 encoded audio"""
    try:
        lang = SUPPORTED_LANGUAGES.get(lang_code, {}).get('gtts_lang', 'en')
        tts = gTTS(text=text, lang=lang, slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
        
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        os.unlink(tmp_file.name)
        return audio_base64
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

def get_saved_images():
    """Get list of saved prediction images"""
    images = []
    upload_folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_folder):
        for filename in os.listdir(upload_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                images.append({
                    'filename': filename,
                    'path': f'/static/uploads/{filename}',
                    'upload_time': datetime.fromtimestamp(
                        os.path.getctime(os.path.join(upload_folder, filename))
                    ).strftime('%Y-%m-%d %H:%M:%S')
                })
    return sorted(images, key=lambda x: x['upload_time'], reverse=True)

@app.route('/')
def index():
    saved_images = get_saved_images()
    return render_template('index.html', saved_images=saved_images)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Get language from request, default to English
    language = request.form.get('language', 'en')
    if language not in SUPPORTED_LANGUAGES:
        language = 'en'
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read and process image
            image = Image.open(file.stream).convert('RGB')
            
            # Save uploaded image with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(file.filename)
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{timestamp}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            
            # Make prediction
            predicted_class, confidence, top_predictions, guidance_dict = predict_traffic_sign(image)
            
            if predicted_class is None:
                return jsonify({'error': 'Model not available'}), 500
            
            # Get guidance in the selected language, fallback to English
            guidance = guidance_dict.get(language, guidance_dict['en'])
            
            # Generate alert message with guidance
            alert_message = f"{predicted_class}. {guidance}"
            audio_base64 = text_to_speech(alert_message, language)
            
            response = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'guidance': guidance,
                'top_predictions': top_predictions,
                'image_url': f'/static/uploads/{filename}',
                'image_filename': filename,
                'audio_data': audio_base64,
                'alert_message': alert_message,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'language': language
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/clear', methods=['POST'])
def clear_predictions():
    """Clear all saved predictions"""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_folder):
            for filename in os.listdir(upload_folder):
                file_path = os.path.join(upload_folder, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        return jsonify({'success': 'All predictions cleared'})
    except Exception as e:
        return jsonify({'error': f'Error clearing predictions: {str(e)}'}), 500

@app.route('/delete_image/<filename>', methods=['DELETE'])
def delete_image(filename):
    """Delete a specific image"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if os.path.exists(filepath):
            os.unlink(filepath)
            return jsonify({'success': 'Image deleted'})
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error deleting image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)