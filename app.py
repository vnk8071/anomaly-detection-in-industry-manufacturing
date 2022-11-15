from flask import Flask, request 
import cv2 
import os
import shutil
import logging
import warnings
import datetime
import zipfile
import tempfile
from omegaconf import OmegaConf
import pandas as pd
from flask import render_template, request, flash, redirect, url_for, Blueprint
from wtforms.validators import ValidationError
from flask_login import login_user, login_required, current_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash
from __init__ import create_app
from models import AnomalyDatabase, User
from database import db, drop_table

from script_inference import get_args, get_inferencer, inference 
from pytorch_lightning import Trainer
from custom_inference import TorchInferencer
from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger

logger = logging.getLogger("anomalib")

user_select = [file.rsplit("/")[-1].split(".")[0] for file in os.listdir("models/")]

app = create_app()
app.app_context().push()
db.create_all()

def time_save():
    now = datetime.datetime.now()
    return now.strftime("%d%m%Y_%H%M%S")

@app.route("/")
def index():
    return render_template("login.html")

@app.route("/index")
@login_required
def home_index():
    return render_template("index.html", name=current_user.name)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        name = request.form.get('name')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        user = User.query.filter_by(name=name).first()

        if not user:
            flash('Please sign up before!')
            return redirect(url_for('signup'))
        elif not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))
        login_user(user, remember=remember)
        return redirect(url_for('home_index'))

@app.route('/signup', methods=['GET', 'POST'])  # we define the sign up path
def signup():  # define the sign up function
    if request.method == 'GET':  # If the request is GET we return the sign up page and forms
        return render_template('signup.html')
    else:  # if the request is POST, then we check if the email doesn't already exist and then we save data
        name = request.form.get('name')
        password = request.form.get('password')
        # if this returns a user, then the email already exists in database
        user = User.query.filter_by(name=name).first()
        if user:  # if a user is found, we want to redirect back to signup page so user can try again
            flash('Username already exists')
            return redirect(url_for('signup'))
        # create a new user with the form data. Hash the password so the plaintext version isn't saved.
        new_user = User(name=name, password=generate_password_hash(
            password, method='sha256'))
        # add the new user to the database
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))


@app.route('/logout')  # define logout path
@login_required
def logout():  # define the logout function
    logout_user()
    return redirect(url_for('home_index'))


@app.route('/train', methods=['GET', 'POST'])
def training():    
    if request.method == 'POST':
        if not request.files["zip_input"]:
            flash('Please attach file for training', 'warning')
            return render_template("train.html", name=current_user.name)
        else:
            file_zip = request.files['zip_input']
            file_name = file_zip.filename.rsplit('.')[0]
            zip_upload_path = f'static/uploads/{file_zip.filename}'
            file_zip.save(zip_upload_path)
            with zipfile.ZipFile(zip_upload_path, 'r') as zip_ref:
                zip_ref.extractall("./data")
            config = get_configurable_parameters(model_name="patchcore", config_path="configs/custom.yaml")
            config['dataset']['category'] = file_name
            save_yaml = OmegaConf.create(config)
            with tempfile.NamedTemporaryFile() as fp:
                OmegaConf.save(config=save_yaml, f="configs/" + file_name + ".yaml")
            datamodule = get_datamodule(config)
            model = get_model(config)
            experiment_logger = get_experiment_logger(config)
            callbacks = get_callbacks(config)

            trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
            logger.info("Training the model.")
            trainer.fit(model=model, datamodule=datamodule)

            logger.info("Loading the best model weights.")
            load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
            trainer.callbacks.insert(0, load_model_callback)

            logger.info("Testing the model.")
            trainer.test(model=model, datamodule=datamodule)

            # Move weight to models folder
            src_dir = f"results/patchcore/mvtec/{file_name}/weights/model.ckpt"
            dst_dir = f"models/{file_name}.ckpt"
            shutil.copy(src_dir,dst_dir)
            flash('Training successful!')
    return render_template("train.html", name=current_user.name, user_select=user_select)

@app.route("/inference", methods=(["GET", "POST"]))
def inference_input():
    if request.method == "POST" and current_user:
        if not request.files["image_input"] or not request.form["user_category"]:
            flash('Please check file and category', 'warning')
            return render_template("inference.html", name=current_user.name)
        else:
            now = time_save()
            image_input = request.files['image_input']
            file_name = image_input.filename.rsplit('.')[0]
            image_upload_path = f'static/uploads/{now}_{file_name}.jpg'
            image_input.save(image_upload_path)
            user_category = request.form["user_category"]
            
            image = cv2.imread(image_upload_path)
            inferencer = TorchInferencer(config="configs/" + user_category + ".yaml", model_source="models/" + user_category + ".ckpt")
            predictions = inferencer.predict(image=image)
            image_predict_path = f"static/uploads/{now}_{file_name}_heatmap.jpg"
            cv2.imwrite(image_predict_path, predictions.heat_map)
            
            result = AnomalyDatabase(image_upload=image_upload_path,
                                anomaly_score=predictions.pred_score,
                                target=predictions.pred_label,
                                image_predict=image_predict_path)
            db.session.add(result)
            db.session.commit()
            return render_template("result_inference.html", name=current_user.name, image_upload=image_upload_path, anomaly_score=predictions.pred_score, target=predictions.pred_label, image_predict=image_predict_path)
    return render_template("inference.html", name=current_user.name, user_select=user_select)

@app.route("/database", methods=(["GET", "POST"]))
def database():
    return render_template("database.html", results=AnomalyDatabase.query.all(), name=current_user.name)

@app.route('/delete/<int:id>')
def delete(id):
    id_delete = AnomalyDatabase.query.get_or_404(id)
    db.session.delete(id_delete)
    db.session.commit()
    return redirect(url_for('database'))

@app.route("/document")
def document():
    return render_template("document.html", name=current_user.name)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000", debug=True)   