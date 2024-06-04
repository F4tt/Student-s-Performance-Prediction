from flask import Flask,render_template,request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')   
def home():
    return render_template('index.html')


@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == "POST":
        try:
            Toan_1_10 = float(request.form['Toan_1_10'])
            Toan_2_10 = float(request.form['Toan_2_10'])
            Toan_1_11 = float(request.form['Toan_1_11'])
            Toan_2_11 = float(request.form['Toan_2_11'])

            Van_1_10 = float(request.form['Van_1_10'])
            Van_2_10 = float(request.form['Van_2_10'])
            Van_1_11 = float(request.form['Van_1_11'])
            Van_2_11 = float(request.form['Van_2_11'])

            Ly_1_10 = float(request.form['Ly_1_10'])
            Ly_2_10 = float(request.form['Ly_2_10'])
            Ly_1_11 = float(request.form['Ly_1_11'])
            Ly_2_11 = float(request.form['Ly_2_11'])

            Anh_1_10 = float(request.form['Anh_1_10'])
            Anh_2_10 = float(request.form['Anh_2_10'])
            Anh_1_11 = float(request.form['Anh_1_11'])
            Anh_2_11 = float(request.form['Anh_2_11'])

            Su_1_10 = float(request.form['Su_1_10'])
            Su_2_10 = float(request.form['Su_2_10'])
            Su_1_11 = float(request.form['Su_1_11'])
            Su_2_11 = float(request.form['Su_2_11'])

            Dia_1_10 = float(request.form['Dia_1_10'])
            Dia_2_10 = float(request.form['Dia_2_10'])
            Dia_1_11 = float(request.form['Dia_1_11'])
            Dia_2_11 = float(request.form['Dia_2_11'])

            Sinh_1_10 = float(request.form['Sinh_1_10'])
            Sinh_2_10 = float(request.form['Sinh_2_10'])
            Sinh_1_11 = float(request.form['Sinh_1_11'])
            Sinh_2_11 = float(request.form['Sinh_2_11'])

            Hoa_1_10 = float(request.form['Hoa_1_10'])
            Hoa_2_10 = float(request.form['Hoa_2_10'])
            Hoa_1_11 = float(request.form['Hoa_1_11'])
            Hoa_2_11 = float(request.form['Hoa_2_11'])

            pred_arg = [Toan_1_10,Van_1_10,Ly_1_10,Hoa_1_10,Sinh_1_10,Su_1_10,Dia_1_10,Anh_1_10
                        ,Toan_2_10,Van_2_10,Ly_2_10,Hoa_2_10,Sinh_2_10,Su_2_10,Dia_2_10,Anh_2_10
                        ,Toan_1_11,Van_1_11,Ly_1_11,Hoa_1_11,Sinh_1_11,Su_1_11,Dia_1_11,Anh_1_11
                        ,Toan_2_11,Van_2_11,Ly_2_11,Hoa_2_11,Sinh_2_11,Su_2_11,Dia_2_11,Anh_2_11]
            
            pred_arg_arr = np.array(pred_arg)
            pred_arg_arr = pred_arg_arr.reshape(1,-1)
            LR = open("LinearRegressionModel.pkl", "rb")
            ml_model = joblib.load(LR)
            model_prediction = ml_model.predict(pred_arg_arr).round(1)

            ChonKhoi_A00 =(model_prediction[:,1] + model_prediction[:,3] + model_prediction[:,4]).astype('float64').round(1)
            ChonKhoi_A01 =(model_prediction[:,1] + model_prediction[:,3] + model_prediction[:,8]).astype('float64').round(1)
            ChonKhoi_B00 =(model_prediction[:,1] + model_prediction[:,4] + model_prediction[:,5]).astype('float64').round(1)
            ChonKhoi_C00 =(model_prediction[:,2] + model_prediction[:,6] + model_prediction[:,7]).astype('float64').round(1)
            ChonKhoi_D00 =(model_prediction[:,1] + model_prediction[:,2] + model_prediction[:,8]).astype('float64').round(1)
        except ValueError:
            return "Vui lòng nhập đầy đủ thông tin"
    return render_template('predict.html',
            kq_Toan = model_prediction[:,1],
            kq_Van = model_prediction[:,2],
            kq_Ly = model_prediction[:,3],
            kq_Hoa = model_prediction[:,4],
            kq_Sinh = model_prediction[:,5],
            kq_Su = model_prediction[:,6],
            kq_Dia = model_prediction[:,7],
            kq_Anh = model_prediction[:,8],
            kq_Total = model_prediction[:,0],
            A00 = ChonKhoi_A00,
            A01 = ChonKhoi_A01,
            B00 = ChonKhoi_B00,
            C00 = ChonKhoi_C00,
            D00 = ChonKhoi_D00
            )

if __name__ == "__main__":
    app.run(host="0.0.0.0")