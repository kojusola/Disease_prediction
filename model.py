import uvicorn
from fastapi import FastAPI
from symptoms import symptom
from typing import Any,Dict
import numpy as np
import pickle
import pandas as pd
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = pickle.load(open('classifier.pkl', 'rb'))

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/predict')
def predict_illness(data:symptom):
    data = data.dict()
    itching=data['itching']
    skin_rash=data['skin_rash']
    nodal_skin_eruptions=data['nodal_skin_eruptions']
    continuous_sneezing=data['continuous_sneezing']
    shivering=data['shivering']
    chills=data['chills']
    joint_pain=data['joint_pain']
    stomach_pain=data['stomach_pain']
    acidity=data['acidity']
    ulcers_on_tongue=data['ulcers_on_tongue']
    muscle_wasting=data['muscle_wasting']
    vomiting=data['vomiting']
    burning_micturition =data['burning_micturition']
    spotting_urination=data['spotting_urination']
    fatigue=data['fatigue']
    weight_gain=data['weight_gain']
    anxiety=data['anxiety']
    cold_hands_and_feets=data['cold_hands_and_feets']
    mood_swings=data['mood_swings']
    weight_loss=data['weight_loss']
    restlessness=data['restlessness']
    lethargy=data['lethargy']
    patches_in_throat=data['patches_in_throat']
    irregular_sugar_level=data['irregular_sugar_level']
    cough=data['cough']
    high_fever=data['high_fever']
    sunken_eyes=data['sunken_eyes']
    breathlessness=data['breathlessness']
    sweating=data['sweating']
    dehydration=data['dehydration']
    indigestion=data['indigestion']
    headache=data['headache']
    yellowish_skin=data['yellowish_skin']
    dark_urine=data['dark_urine']
    nausea=data['nausea']
    loss_of_appetite=data['loss_of_appetite']
    pain_behind_the_eyes=data['pain_behind_the_eyes']
    back_pain=data['back_pain']
    constipation=data['constipation']
    abdominal_pain=data['abdominal_pain']
    diarrhoea=data['diarrhoea']
    mild_fever=data['mild_fever']
    yellow_urine=data['yellow_urine']
    yellowing_of_eyes=data['yellowing_of_eyes']
    acute_liver_failure=data['acute_liver_failure']
    fluid_overload=data['fluid_overload'] 
    swelling_of_stomach=data['swelling_of_stomach']
    swelled_lymph_nodes=data['swelled_lymph_nodes']
    malaise=data['malaise']
    blurred_and_distorted_vision=data['blurred_and_distorted_vision']
    phlegm=data['phlegm']
    throat_irritation=data['throat_irritation']
    redness_of_eyes=data['redness_of_eyes']
    sinus_pressure=data['sinus_pressure']
    runny_nose=data['runny_nose']
    congestion=data['congestion']
    chest_pain=data['chest_pain']
    weakness_in_limbs=data['weakness_in_limbs'] 
    fast_heart_rate=data['fast_heart_rate']
    pain_during_bowel_movements=data['pain_during_bowel_movements']
    pain_in_anal_region=data['pain_in_anal_region']
    bloody_stool=data['bloody_stool']
    irritation_in_anus=data['irritation_in_anus']
    neck_pain=data['neck_pain']
    dizziness=data['dizziness']
    cramps=data['cramps']
    bruising=data['bruising']
    obesity=data['obesity']
    swollen_legs=data['swollen_legs']
    swollen_blood_vessels=data['swollen_blood_vessels']
    puffy_face_and_eyes=data['puffy_face_and_eyes']
    enlarged_thyroid=data['enlarged_thyroid']
    brittle_nails=data['brittle_nails']
    swollen_extremeties=data['swollen_extremeties']
    excessive_hunger=data['excessive_hunger']
    extra_marital_contacts=data['extra_marital_contacts']
    drying_and_tingling_lips=data['drying_and_tingling_lips']
    slurred_speech=data['slurred_speech']
    knee_pain=data['knee_pain']
    hip_joint_pain=data['hip_joint_pain']
    muscle_weakness=data['muscle_weakness']
    stiff_neck=data['stiff_neck']
    swelling_joints=data['swelling_joints']
    movement_stiffness=data['movement_stiffness']
    spinning_movements=data['spinning_movements']
    loss_of_balance=data['loss_of_balance']
    unsteadiness=data['unsteadiness']
    weakness_of_one_body_side=data['weakness_of_one_body_side']
    loss_of_smell=data['loss_of_smell']
    bladder_discomfort=data['bladder_discomfort']
    foul_smell_of_urine=data['foul_smell_of_urine']
    continuous_feel_of_urine=data['continuous_feel_of_urine']
    passage_of_gases=data['passage_of_gases']
    internal_itching=data['internal_itching']
    toxic_look_typhos=data['toxic_look_typhos'] 
    depression=data['depression']
    irritability=data['irritability']
    muscle_pain=data['muscle_pain']
    altered_sensorium=data['altered_sensorium']
    red_spots_over_body= data['red_spots_over_body']
    belly_pain= data['belly_pain']
    abnormal_menstruation=data['abnormal_menstruation']
    dischromic_patches=data['dischromic_patches']
    watering_from_eyes=data['watering_from_eyes']
    increased_appetite=data['increased_appetite']
    polyuria=data['polyuria']
    family_history=data['family_history']
    mucoid_sputum=data['mucoid_sputum']
    rusty_sputum=data['rusty_sputum']
    lack_of_concentration=data['lack_of_concentration']
    visual_disturbances=data['visual_disturbances']
    receiving_blood_transfusion=data['receiving_blood_transfusion']
    receiving_unsterile_injections=data['receiving_unsterile_injections']
    coma=data['coma']
    stomach_bleeding=data['stomach_bleeding']
    distention_of_abdomen=data['distention_of_abdomen']
    history_of_alcohol_consumption=data['history_of_alcohol_consumption']
    fluid_overload=data['fluid_overload']
    blood_in_sputum=data['blood_in_sputum']
    prominent_veins_on_calf=data['prominent_veins_on_calf']
    palpitations=data['palpitations'] 
    painful_walking=data['painful_walking']
    pus_filled_pimples=data['pus_filled_pimples']
    blackheads=data['blackheads']
    scurring=data['scurring']
    skin_peeling=data['skin_peeling']
    silver_like_dusting=data['silver_like_dusting']
    small_dents_in_nails=data['small_dents_in_nails']
    inflammatory_nails=data['inflammatory_nails']
    blister=data['blister']
    red_sore_around_nose=data['red_sore_around_nose']
    yellow_crust_ooze=data['yellow_crust_ooze']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    bam =[[itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering, chills, joint_pain, stomach_pain, acidity, ulcers_on_tongue,muscle_wasting, vomiting, burning_micturition, spotting_urination, fatigue, weight_gain, anxiety, cold_hands_and_feets, mood_swings, weight_loss, restlessness, lethargy, patches_in_throat, irregular_sugar_level, cough, high_fever, sunken_eyes, breathlessness, sweating, dehydration,indigestion, headache, yellowish_skin, dark_urine, nausea, loss_of_appetite, pain_behind_the_eyes, back_pain, constipation, abdominal_pain, diarrhoea, mild_fever, yellow_urine, yellowing_of_eyes,acute_liver_failure, fluid_overload, swelling_of_stomach, swelled_lymph_nodes, malaise,blurred_and_distorted_vision, phlegm, throat_irritation, redness_of_eyes, sinus_pressure,runny_nose, congestion, chest_pain, weakness_in_limbs, fast_heart_rate, pain_during_bowel_movements, pain_in_anal_region, bloody_stool, irritation_in_anus, neck_pain, dizziness, cramps, bruising, obesity, swollen_legs, swollen_blood_vessels, puffy_face_and_eyes, enlarged_thyroid, brittle_nails, swollen_extremeties, excessive_hunger, extra_marital_contacts, drying_and_tingling_lips, slurred_speech, knee_pain, hip_joint_pain, muscle_weakness, stiff_neck, swelling_joints, movement_stiffness, spinning_movements, loss_of_balance, unsteadiness, weakness_of_one_body_side, loss_of_smell, bladder_discomfort, foul_smell_of_urine, continuous_feel_of_urine, passage_of_gases, internal_itching, toxic_look_typhos, depression, irritability, muscle_pain, altered_sensorium, red_spots_over_body, belly_pain, abnormal_menstruation, dischromic_patches, watering_from_eyes, increased_appetite, polyuria, family_history, mucoid_sputum, rusty_sputum, lack_of_concentration, visual_disturbances, receiving_blood_transfusion, receiving_unsterile_injections, coma, stomach_bleeding, distention_of_abdomen, history_of_alcohol_consumption, fluid_overload, blood_in_sputum, prominent_veins_on_calf, palpitations, painful_walking, pus_filled_pimples, blackheads, scurring, skin_peeling, silver_like_dusting, small_dents_in_nails,inflammatory_nails, blister, red_sore_around_nose, yellow_crust_ooze]]
    
    y_pred_m1 = classifier.predict_proba(bam)
    best_three1 = np.argsort(-y_pred_m1, axis=1)[:,:3]
    predictions= classifier.classes_[best_three1]
    return predictions[0][0],predictions[0][1],predictions[0][2]
    

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)