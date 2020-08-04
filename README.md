# Gender Recognition by Voice
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/TRoboto/datacamp-downloader/blob/master/LICENSE)

## Table of Contents
- [Gender Recognition by Voice](#gender-recognition-by-voice)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Instructions](#instructions)
    - [Installation](#installation)
    - [How to use](#how-to-use)
	
## Description
The aim of this project is to develop a model that can take a waveform as its input and output speaker’s gender, whether male or female. The developed model is trained on the Common Voice dataset, which can be found [here](https://www.kaggle.com/mozillaorg/common-voice). It includes more than 70,000 records of male and female from different countries.  

The project includes a flash app that eases the use of the final model. The final model, XGBoost, managed to achieve an accuracy of 97.1% on the test set, which is almost perfect. 

The code used for training can be found in the `src` folder.  

For more details about the project and the procedure I followed to develop this model, **refer to `report.pdf`, which can be found in the main repo.**

**NOTE :** If you want to train and test models, you can use the features I extracted from the dataset from [here](https://drive.google.com/drive/folders/1kVCoOcnIJgV4fChpoEG7iandE6bM8AHt?usp=sharing). Don't forget to put them in `src/results`

**Support!**  
If you find this project helpful, please support me by starring this repository.

## Instructions

### Installation
1. Download this repository or clone it using:
```
git clone https://github.com/TRoboto/Gender-Recognition-by-Voice
```
2. Change the current working directory to the code location, run:
```
cd Gender-Recognition-by-Voice/src
```
3. Download the required dependencies, run:
```
pip install -r requirements.txt
```
4. Install FFmpeg library:

	- For Anaconda:
	```
	conda install -c conda-forge ffmpeg
	```
	- Others:
	```
	pip install ffmpeg
	```
	
### How to use

1. Start flask localhost server, run:
```
python app.py
```
You should see something like this:
```
* Serving Flask app "app" (lazy loading)
* Environment: production
WARNING: This is a development server. Do not use it in a production deployment.
Use a production WSGI server instead.
* Debug mode: off
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

2. Open the server in the browser like this:
```
http://127.0.0.1:5000/predict?audio_file=file_location
```
Replace `file_location` with the location of an audio file on your pc like this:
```
http://127.0.0.1:5000/predict?audio_file=C:\my_voice.m4a
```
**NOTE:** You can use Voice Recorder app to record five seconds of your voice and then, save it and test it.

3. If everything is working as expecting, you should see the output as follows:
```
{
"response":{
	"audio_file":"C:\\my_voice.m4a",
	"female":"0.9750009160488844",
	"male":"0.024999084",
	"time_taken":"0.6616175174713135"
	}
}

```
where the value of `female`/`male` is the probability of being female/male. 
```
the probability of male = 1 - the probability of female
```
