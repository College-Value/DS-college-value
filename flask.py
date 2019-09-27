from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import sys

flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Income Estimator", 
		  description = "Predict income 6 years after enrolling in college")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
				  {'STABBR': fields.String(required = True, 
				  							   description="STABBR",),
    					  				 	#    help="Sepal Length cannot be blank"),
				  'Major': fields.String(required = True, 
				  							   description="Major",),
    					  				 	#    help="Sepal Width cannot be blank"),
				  'Age': fields.Float(required = True, 
				  							description="Age",),
    					  				 	# help="Petal Length cannot be blank"),
				  'Race': fields.String(required = True, 
				  							description="Race",),
    					  				 	# help="Petal Width cannot be blank")}
                  'Born_in_US': fields.String(required = True, 
				  							description="Born in US",),
                  'Median_HH_inc': fields.String(required = True,
                                            description="Median household income")
                  }
)

predictor = joblib.load('college_pipeline_local.joblib')

start_input_df = pd.DataFrame(columns=['STABBR', 'Agriculture', 'Conservation', 'Architecture',
    'Gender Studies', 'Journalism', 'Communications', 'IST',
    'Culinary Services', 'Education', 'Engineering',
    'Engineering Technology', 'Foreign Languages',
    'Family and Consumer Science', 'Legal Studies', 'English',
    'Liberal Arts', 'Library Science', 'Biomedical Science', 'Mathematics',
    'Military Science', 'Interdisciplinary Studies', 'Parks and Recreation',
    'Philosophy', 'Reilgious Studies', 'Physics', 'Science Technologies',
    'Psychology', 'Law Enforcement', 'Public Administration',
    'Social Science', 'Construction Trades', 'Technician',
    'Precision Production', 'Transportation And Materials Moving',
    'Visual And Performing Arts', 'Health Studies',
    'Business Management and Marketing', 'History', 'AGEGE24', 'PCT_WHITE',
    'PCT_BLACK', 'PCT_ASIAN', 'PCT_HISPANIC', 'PCT_BORN_US',
    'MEDIAN_HH_INC'],
    data=[['AL', 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0.079999998, 0.6, 0, 0, 0, 94.73999786, 49720.22]])

majors = ['Agriculture', 'Conservation', 'Architecture', 'Gender Studies', 
        'Journalism', 'Communications', 'IST', 'Culinary Services', 
        'Education', 'Engineering', 'Engineering Technology', 
        'Foreign Languages', 'Family and Consumer Science', 'Legal Studies',
        'English', 'Liberal Arts', 'Library Science', 'Biomedical Science',
        'Mathematics', 'Military Science', 'Interdisciplinary Studies', 
        'Parks and Recreation', 'Philosophy', 'Reilgious Studies', 'Physics', 
        'Science Technologies', 'Psychology', 'Law Enforcement', 
        'Public Administration', 'Social Science', 'Construction Trades', 
        'Technician', 'Precision Production', 
        'Transportation And Materials Moving', 'Visual And Performing Arts', 
        'Health Studies', 'Business Management and Marketing', 'History'
]

races = ['PCT_WHITE', 'PCT_BLACK', 'PCT_ASIAN', 'PCT_HISPANIC']

def set_major(df, major_chosen):
    df = df.copy()
    for major in majors:
        df[major] = 0
    df[major_chosen] = 1
    return df

def set_race(df, race_input):
    df = df.copy()
    for race in races:
        df[race] = 0
    df[race_input] = 1
    return df

def set_born_us(df, born_US):
    df = df.copy()
    df['PCT_BORN_US'] = born_US
    return df

def set_STABBR(df, STABBR):
    df = df.copy()
    df['STABBR'] = STABBR
    return df

def set_AGEGE24(df, age):
    df = df.copy()
    if age > 23:
        df.AGEGE24 = 1
    else:
        df.AGEGE24 = 0
    return df

def set_MEDIAN_HH_INC(df, MEDIAN_HH_INC):
    df = df.copy()
    df.MEDIAN_HH_INC = MEDIAN_HH_INC
    return df

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		try: 
            input_df = set_major(start_input_df, request.json['Major'])
            input_df = set_race(input_df, request.json['Race'])
            input_df = set_AGEGE24(input_df, request.json['Age'])
            input_df = set_STABBR(input_df, request.json['STABBR'])
            input_df = set_born_us(input_df, request.json['Born_in_US'])
            input_df = set_MEDIAN_HH_INC(df, request.json['Median_HH_inc'])
			prediction = predictor.predict(input_df)
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": prediction[0]
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})
