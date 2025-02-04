import gradio as gr
import pandas as pd
import joblib
import os

current_path = os.path.dirname(os.path.abspath(__file__))

# Load Modul yang sudah dilatih
model_knn = joblib.load(open('trained_model/model_knn_trained.joblib', 'rb')) #Akurasi: 79,13%
model_nb = joblib.load(open('trained_model/model_nb_trained.joblib', 'rb')) #Akurasi: 79,64%
model_lr = joblib.load(open('trained_model/model_lr_trained.joblib', 'rb'))  #Akurasi: 80,41%
model_svm = joblib.load(open('trained_model/model_svm_trained.joblib', 'rb')) #Akurasi: 85,11%

# Fungsi Prediksi
def predict_survival(sex, age, fare, cherbourg, queenstown, southampton, class_1, class_2, class_3, model):
    data = pd.DataFrame({
        'Sex': [sex], 
        'Age': [age], 
        'Passenger Fare': [fare],
        'Port of Embarkation_Cherbourg': [cherbourg],
        'Port of Embarkation_Queenstown': [queenstown],
        'Port of Embarkation_Southampton': [southampton],
        'Passenger Class_First': [class_1],
        'Passenger Class_Second': [class_2],
        'Passenger Class_Third': [class_3],
        })
    
    prediction = model.predict(data)[0]
    probabilitas = model.predict_proba(data)[0]
    prediksi = "SELAMAT" if prediction == 1 else "TIDAK SELAMAT"

    if prediction == 1 :
        stat = f"{prediksi} dengan probabilitas {probabilitas[prediction] * 100:.2f}%"
        img_file = "survived.gif"
        info = "Selamat"
    else:
        stat = f"{prediksi} dengan probabilitas {probabilitas[prediction] * 100:.2f}%"
        img_file = "drowned.gif"
        info = "Tidak Selamat"
    return (stat, img_file, info)

# Frontend Gradio
with gr.Blocks() as app:

    with gr.Row(variant="panel"):
        tittle = gr.HTML("""
                        <center><h2>PRAKTIKUM 4 - BDA | APLIKASI PREDIKSI KAPAL TITANIC</h2>
                        <img src="https://img1.picmix.com/output/pic/normal/3/9/8/4/7514893_a17d6.gif" alt="titanic" width="200" height="200"></center>
                        """)

        info = gr.HTML("""
                       <p align="justify"><h3>
                       Aplikasi ini dapat memprediksi keselamatan seorang penumpang kapal titanic berdasarkan data inputan baru yang diinputkan (sebagai penumpang baru), prediksi diproses menggunakan model machine learning yang sudah dilatih sebelumnya. Berdasarkan dataset "titanic.csv" yang sudah melalui tahap preprocessing sebelumnya.<br>
                       <br>
                       Aplikasi ini dibuat untuk memenuhi tugas <b>"Big Data Analysis"</b>
                       </h3></p>
                       """)

    with gr.Row():
        sex = gr.Dropdown(['Female', 'Male'], value='Female', label="Sex")
        age = gr.Slider(0.1667, 80.0, value=24.0, label="Age")
        fare = gr.Slider(0.0, 512.3292, value=150.0, label="Pasenger Fare")
        port = gr.Dropdown(['Cherbourg', 'Queenstown', 'Southampton'], value='Cherbourg', label="Port of Embarkation")
        p_class = gr.Dropdown(['Fist Class', 'Second Class', 'Third Class'], value="Fist Class", label="Passenger Class")
    
    # UNTUK DEBUGGING
    # =================
    # with gr.Row():
    #     input_output = gr.Textbox(label="Input")

    # Button prediksi
    with gr.Row():
        button_submit = gr.Button("Kalkulasi Prediksi", variant="primary")
    
    # Hasil Prediksi
    with gr.Row():
        with gr.Column():
            with gr.Group():
                tittle = gr.HTML("<center>Prediksi Model K-NN | Akurasi Model: 79,13%</>")
                knn_img = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, height=126)
                knn_output = gr.Textbox(lines=1, show_label=False)
        with gr.Column():
            with gr.Group():
                tittle = gr.HTML("<center>Prediksi Model Naive Bayes | Akurasi Model: 79,64%</>")
                nb_img = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, height=126)
                nb_output = gr.Textbox(lines=1, show_label=False)
    with gr.Row():
        with gr.Column():
            with gr.Group():
                tittle = gr.HTML("<center>Prediksi Model Logistic Regression | Akurasi Model: 80,41%</>")
                lr_img = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, height=126)
                lr_output = gr.Textbox(lines=1, show_label=False)
        with gr.Column():
            with gr.Group():
                tittle = gr.HTML("<center>Prediksi Model SVM | Akurasi Model: 85,11%</>")
                svm_img = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, height=126)
                svm_output = gr.Textbox(lines=1, show_label=False)
    

    # Update textbox input_output ketika pilihan inputan berubah
    inputs = [sex, age, fare, port, p_class]

    # Fungsi untuk mengubah jenis kelamin
    def jenis_kelamin(s):
        kelamin = s
        return 1 if s == "Male" else 0
    def port_of_embarkation(pe):
        if pe == "Cherbourg":
            return 1, 0, 0
        elif pe == "Queenstown":
            return 0, 1, 0
        else:
            return 0, 0, 1
    def passenger_class(pc):
        if pc == "Fist Class":
            return 1, 0, 0
        elif pc == "Second Class":
            return 0, 1, 0
        else:
            return 0, 0, 1
    
    # UNTUK DEBUGGING
    # =================
    # for input in inputs:
    #     input.change(
    #         lambda s, a, f, pe, pc: (
    #             f"Sex: {jenis_kelamin(s)} | "
    #             f"Age: {a} | "
    #             f"Passenger Fare: {f} | "
    #             f"Port of Embarkation: {port_of_embarkation(pe)} | "
    #             f"Passenger Class: {passenger_class(pc)}"
    #         ),
    #         inputs=inputs,
    #         outputs=input_output
    #     )
    
    # Update image output ketika button diklik
    button_submit.click(lambda s, a, f, pe, pc: predict_survival(jenis_kelamin(s), a, f, *port_of_embarkation(pe), *passenger_class(pc), model_knn)[1], inputs, knn_img)
    button_submit.click(lambda s, a, f, pe, pc: predict_survival(jenis_kelamin(s), a, f, *port_of_embarkation(pe), *passenger_class(pc), model_nb)[1], inputs, nb_img)
    button_submit.click(lambda s, a, f, pe, pc: predict_survival(jenis_kelamin(s), a, f, *port_of_embarkation(pe), *passenger_class(pc), model_lr)[1], inputs, lr_img)
    button_submit.click(lambda s, a, f, pe, pc: predict_survival(jenis_kelamin(s), a, f, *port_of_embarkation(pe), *passenger_class(pc), model_svm)[1], inputs, svm_img)

    # Update textbox output ketika button diklik
    button_submit.click(lambda s, a, f, pe, pc: predict_survival(jenis_kelamin(s), a, f, *port_of_embarkation(pe), *passenger_class(pc), model_knn)[0], inputs, knn_output)
    button_submit.click(lambda s, a, f, pe, pc: predict_survival(jenis_kelamin(s), a, f, *port_of_embarkation(pe), *passenger_class(pc), model_nb)[0], inputs, nb_output)
    button_submit.click(lambda s, a, f, pe, pc: predict_survival(jenis_kelamin(s), a, f, *port_of_embarkation(pe), *passenger_class(pc), model_lr)[0], inputs, lr_output)
    button_submit.click(lambda s, a, f, pe, pc: predict_survival(jenis_kelamin(s), a, f, *port_of_embarkation(pe), *passenger_class(pc), model_svm)[0], inputs, svm_output)

app.launch(share=False, allowed_paths=[current_path])
