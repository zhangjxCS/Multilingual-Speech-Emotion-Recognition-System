import os

if __name__ == "__main__":

    emotions_map = {
            'Neutral': 'neutral',
            'Anger': 'angry', 
            'Happiness': 'happy', 
            'Sadness': 'sad', 
            'Fear': 'fear',
            'Disgust': 'disgust'
    }
    dataset_location = './MESD_Final_Splits/'

    for split in range(5):
        print('SPLIT:', split)
        split_location = dataset_location + 'split_' + str(split)
        emotion_counter = {}

        for audiofile in os.listdir(split_location):
            emotion = emotions_map[audiofile.split("_")[0]]
        
            #Emotion counter
            if emotion not in emotion_counter:
                emotion_counter[emotion] = 0
            emotion_counter[emotion] += 1


        for emotion in emotion_counter:
            print('--', emotion, ':', emotion_counter[emotion])

        print('TOTAL:', sum(emotion_counter.values()))

        print('-'*30)


