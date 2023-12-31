import os
from flask import Flask, request, jsonify
from facial_expression import get_dominant_emotions,recognize_expression_video,get_video_time,measure_stress,blink_calculator,stress_analysis,count_head_movements

app = Flask(__name__)
    
@app.route('/uploadVideo', methods=['POST'])
def uploadVideo():

    try:
        file = request.files['Video']
        filename = file.filename
        video_path = os.path.join("uploadedVideos", filename)
        file.save(video_path)
        [emotionframes,expressionPercentage]= recognize_expression_video(video_path)
        dominantEmotionEachSecond=get_dominant_emotions(emotionframes,video_path)
        #stress_level=measure_stress(video_path)
        blinks = round(blink_calculator(video_path) / get_video_time(video_path), 2)
        head_mouvements= count_head_movements(video_path)
        #conc=stress_analysis(blinks,stress_level)
        return jsonify({"expressions":expressionPercentage,
                        "expressionsTime":dominantEmotionEachSecond,
                        "blinks": blinks, 
                        "head_mouvements":head_mouvements
                        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e)
            }), 400 
    
    
if __name__ == '__main__':
    app.run(debug=True, port=4000)
