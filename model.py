from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch 
import speech_recognition as sr
import io
from pydub import AudioSegment

# print(sr.Microphone.list_microphone_names()) # testing your mic's on your pc

checkout = "yongjian/wav2vec2-large-a"

tokenizer = Wav2Vec2Processor.from_pretrained(checkout)
model = Wav2Vec2ForCTC.from_pretrained(checkout)

r = sr.Recognizer()

with sr.Microphone(sample_rate=16000) as source:
    print('You can start speaking now...')
    while True:
        audio = r.listen(source) #pyaudio object
        data = io.BytesIO(audio.get_wav_data()) #list of bytes
        clip = AudioSegment.from_file(data) # numpy array
        x = torch.FloatTensor(clip.get_array_of_samples())
        
        # inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').inputs_values
        inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt', padding = 'longest').input_values

        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis = -1)
        text = tokenizer.batch_decode(tokens)
        text = str(text).lower()
        
        print('you said: ', text)


