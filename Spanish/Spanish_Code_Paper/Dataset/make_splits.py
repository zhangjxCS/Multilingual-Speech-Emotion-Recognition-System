import os
import random
import shutil

pwd

if __name__ == '__main__':

    n_splits = 5
    dataset_location = '/home/rl3155/MESD_All/'
    output_location = './MESD_Final_Splits/'
    os.makedirs(output_location, exist_ok=True)
    emotion_counter = {}
    emotion_to_audio = {}
    emotions_map = {
            'Neutral': 'neutral',
            'Anger': 'angry', 
            'Happiness': 'happy', 
            'Sadness': 'sad', 
            'Fear': 'fear',
            'Disgust': 'disgust'
    }


    for audiofile in os.listdir(dataset_location):
        if audiofile == 'desktop.ini':
            continue
        emotion = emotions_map[audiofile.split('_')[0]] # get emotion for that file
        
        #Emotion counter
        if emotion not in emotion_counter:
            emotion_counter[emotion] = 0
        emotion_counter[emotion] += 1

        #Add files in emotion
        if emotion not in emotion_to_audio:
            emotion_to_audio[emotion] = []
        emotion_to_audio[emotion].append(audiofile)
        
    #Random shuffle the files
    for emotion in emotion_to_audio:
        random.shuffle(emotion_to_audio[emotion])
        random.shuffle(emotion_to_audio[emotion])

    #Make 5 splits
    for emotion in emotion_to_audio:
        print('EMOTION:',emotion)
        for i in range(n_splits):
            split_location = output_location + 'split_' + str(i) + '/'
            os.makedirs(split_location, exist_ok=True)
        
            max_len = len(emotion_to_audio[emotion])
            split_size = max_len // n_splits

            start_range = i * split_size
            end_range =  (i+1) * split_size

            for filename in emotion_to_audio[emotion][start_range : end_range]:
                source_location = dataset_location + filename
                target_location = split_location + filename
                shutil.copyfile(source_location, target_location)

        #randomly place remaining files in different splits
        for filename in emotion_to_audio[emotion][end_range : max_len]:
            selected_split = random.sample([0,1,2,3,4], 1)[0]
            source_location = dataset_location + filename
            target_location = output_location + 'split_' + str(selected_split) + '/' + filename
            shutil.copyfile(source_location, target_location)



