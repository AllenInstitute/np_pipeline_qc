from lims_validation import run_validation
from copy_d2_lims_files_for_upload import transfer_session
from validate_local_d2_files_ccb import validate_d2_files
import get_sessions as gs
import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette, QColor
import cv2
import os, glob
import subprocess
from datetime import datetime
import json

# sources = [r"\\10.128.50.43\sd6.3", 
#            r"\\10.128.50.20\sd7", r"\\10.128.50.20\sd7.2", 
#            r"\\10.128.54.20\sd8", r"\\10.128.54.20\sd8.2", r"\\10.128.54.20\sd8.3",
#            r"\\10.128.54.19\sd9"
#            ]
source_volume_config = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\source_list.json"
with open(source_volume_config, 'r') as f:
    sources = json.load(f)
    
exe_relative_path = r"c\Program Files\AIBS_MPE\createDay2\createDay2.exe"
acq_to_sync_comp_dict = {'W10DT05501': r'\\W10DTSM18306',
						 'W10DT05515': r'\\W10DTSM112719'}

def start():
    #QtGui.QApplication.setGraphicsSystem("raster")
	app = QApplication.instance()
	palette = QPalette()
	palette.setColor(QPalette.Window, QColor(53, 53, 53))
	palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
	palette.setColor(QPalette.Base, QColor(25, 25, 25))
	palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
	palette.setColor(QPalette.ToolTipBase, QColor(250, 250, 250))
	palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
	palette.setColor(QPalette.Text, QColor(0, 0, 0))
	palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
	palette.setColor(QPalette.BrightText, QColor(250, 0, 0))
	palette.setColor(QPalette.Link, QColor(42, 130, 218))
	palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
	palette.setColor(QPalette.HighlightedText, QColor(250, 250, 250))
	
	if app is None:
		app = QApplication([])
	app.setPalette(palette)
	D2_obj= D2_validation_tool(app)
	app.exec_()
    
class D2_validation_tool():
    
	def __init__(self, app):

		self.app = app
		self.rig_limsdirectory_dict = {'NP.1': r'\\W10dt05501\e', 'NP.0': r'\\W10dt05515\e'}

		self.sessionID = None
		self.passed_d2_local_validation = None
		self.lims_pass = None
		self.probes_to_run = 'ABCDEF'
		self.acq_computer_name = None
		self.session_directory = None
		self.limsID = None
		self.previousSessionID = 'Session ID'

		self.default_button_background = "background-color: rgb(220, 220, 220)"

		self.mainWin = QMainWindow()
		self.mainWidget = QWidget()
		self.mainWin.setCentralWidget(self.mainWidget)
		self.mainWin.closeEvent = self.closeEvent
		self.mainWin.setWindowTitle('LIMS validation tool')
		self.mainLayout = QGridLayout()

		self.controlPanelLayout = QGridLayout()
		self.mainLayout.addLayout(self.controlPanelLayout, 0, 0)

		self.sessionIDBox = QLineEdit('Session ID')
		self.sessionIDBox.editingFinished.connect(self.set_session)
		self.controlPanelLayout.addWidget(self.sessionIDBox, 0, 0, 1, 1)

		self.probesToRunBox = QLineEdit('ABCDEF')
		self.probesToRunBox.returnPressed.connect(self.set_probes_to_run)
		self.controlPanelLayout.addWidget(self.probesToRunBox, 1, 0, 1, 1)

		self.checkLIMSButton = QPushButton('Check LIMS')
		self.checkLIMSButton.setStyleSheet(self.default_button_background)
		self.checkLIMSButton.clicked.connect(self.check_LIMS)
		self.controlPanelLayout.addWidget(self.checkLIMSButton, 0, 1, 1, 1)

		self.validateLocalFilesButton = QPushButton('Validate files on LIMS upload drive')
		self.validateLocalFilesButton.setStyleSheet(self.default_button_background)
		self.validateLocalFilesButton.clicked.connect(self.validate_d2_files_for_upload)
		self.controlPanelLayout.addWidget(self.validateLocalFilesButton, 0, 2, 1, 1)

		self.copyFilesButton = QPushButton('Copy files to LIMS upload drive')
		self.copyFilesButton.setStyleSheet(self.default_button_background)
		self.copyFilesButton.clicked.connect(self.copy_data_for_d2_lims_upload)
		self.controlPanelLayout.addWidget(self.copyFilesButton, 0, 3, 1, 1)

		self.uploadButton = QPushButton('Initiate Upload to LIMS')
		self.uploadButton.setStyleSheet(self.default_button_background)
		self.uploadButton.clicked.connect(self.initiate_d2_upload)
		self.controlPanelLayout.addWidget(self.uploadButton, 1, 1, 1, 1)

		self.logOutput = QTextEdit()
		self.logOutput.setStyleSheet("color:white")
		self.logOutput.setReadOnly(True)
		self.logOutput.setLineWrapMode(QTextEdit.NoWrap)
		self.controlPanelLayout.addWidget(self.logOutput, 2, 0, 1, 5)

		self.mainWidget.setLayout(self.mainLayout)
		self.mainWin.show()


	def set_session(self):
		self.sessionID = self.sessionIDBox.text()
		if self.previousSessionID != self.sessionID:
			try:
				if os.path.exists(self.sessionID):
					self.logOutput.append('\nSession ID is already a valid path\n'
						'Verifying that it is a valid session directory')
					if gs.validate_session_dir(self.sessionID, [9, 10]):
						self.session_directory = self.sessionID
						self.sessionID = os.path.basename(self.session_directory)
				else:
					self.session_directory = gs.get_sessions(sources, limsID=self.sessionID)
					if len(self.session_directory) == 0:
						self.logOutput.append('\nCould not find session directory for session {}'.format(self.sessionID))
					else:
						self.session_directory = self.session_directory[0]

				self.limsID = self.sessionID.split('_')[0]
				self.logOutput.append('\nset session to {}'.format(self.sessionID))
				self.logOutput.append('found session directory {}'.format(self.session_directory))
				self.sessionID = os.path.basename(self.session_directory)	
				self.reset()
			
			except Exception as e:
				self.logOutput.append('Could not set session to {}, due to error {}'
					.format(self.sessionIDBox.text(), e))

	
	def reset(self):
		self.passed_d2_local_validation = None
		self.lims_pass = None
		self.probes_to_run = 'ABCDEF'
		self.acq_computer_name = None
		self.previousSessionID = self.sessionID

		self.checkLIMSButton.setStyleSheet(self.default_button_background)
		self.validateLocalFilesButton.setStyleSheet(self.default_button_background)


	def set_probes_to_run(self):
		self.probes_to_run = self.probesToRunBox.text()
		self.logOutput.append('Running probes {}'.format(self.probes_to_run))


	def check_LIMS(self):

		try:
			lims_results_save_path = os.path.join(self.session_directory, 'lims_upload_report_'+str(self.limsID)+'.json')
			self.logOutput.append('\nChecking LIMS for session {} \nSaving full results to {}'
				.format(self.limsID, lims_results_save_path))
			lims_validation_results = run_validation(self.limsID, lims_results_save_path)
			
			for lims_upload_day, common_string in zip(['D1_upload_summary', 'D2_upload_summary'], 
				['Raw data', 'Sorted data']):
				if lims_validation_results[lims_upload_day]['upload_exists']:
					upload_pass = lims_validation_results[lims_upload_day]['pass']
					if upload_pass:
						self.logOutput.append(common_string + ' lims upload confirmed!')
					else:
						self.logOutput.append(common_string + ' lims upload failed due to following errors: \n')
						for e in lims_validation_results[lims_upload_day]['errors']:
							self.logOutput.append(e)
						if lims_upload_day == 'D1_upload_summary':
							self.logOutput.append('\nThe raw data for this experiment is not yet in LIMS. \n'
								'Please make sure the raw data is uploaded before proceeding')
							break 

				else:
					self.logOutput.append(common_string + ' lims upload missing \n')

					if lims_upload_day == 'D1_upload_summary':
						self.logOutput.append('The raw data for this experiment is not yet in LIMS.\n'
							'Please make sure the raw data is uploaded before proceeding')
						break
					else:
						self.logOutput.append('Please proceed to upload the sorting data \n')

			self.lims_pass = (lims_validation_results['D1_upload_summary']['pass'] and 
				lims_validation_results['D2_upload_summary']['pass'])

			if self.lims_pass:
				self.checkLIMSButton.setStyleSheet("background-color: rgb(0, 200, 0)")
			else:
				self.checkLIMSButton.setStyleSheet("background-color: rgb(200, 0, 0)")
		except Exception as e:

			self.logOutput.append('Could not check LIMS for session due to error {}'.format(e))


	def validate_d2_files_for_upload(self):
		
		try:
			self.get_acq_computername()
			self.logOutput.append('validating local files for session {}'.format(self.sessionID))
			self.passed_d2_local_validation = validate_d2_files(self.sessionID, self.acq_computer_name)
			if self.passed_d2_local_validation:
				self.logOutput.append('Session {} passed local file validation (all files on E drive)'.format(self.sessionID))
				self.validateLocalFilesButton.setStyleSheet("background-color: rgb(0, 200, 0)")
			else:
				self.logOutput.append('Session {} FAILED local file validation (see console for details)'.format(self.sessionID))
				self.validateLocalFilesButton.setStyleSheet("background-color: rgb(200, 0, 0)")
		except Exception as e:
			self.logOutput.append('Could not validate d2 files for session due to error {}'.format(e))

	
	def get_acq_computername(self):
		computer_name_dict = {'NP.0': 'W10dt05515',
								'NP.1': 'W10dt05501'}
		try:
			platform_file = glob.glob(os.path.join(self.session_directory, '*platform*.json'))[0]
			with open(platform_file, 'r') as file:
				platform_info = json.load(file)

			computer_name = computer_name_dict[platform_info['rig_id']]

		except Exception as e:
			print('Error getting rig ID from platform json: {}, due to error {}' 
	        	.format(os.path.join(self.session_directory, '*platform*.json'), e))
			text, ok = QInputDialog.getText(self.mainWin, 'Please enter ACQ computer name (eg W10dt05515)',
															'Comp name:')
			if ok and text:
				computer_name = text
			else:
				computer_name = None
		
		self.acq_computer_name = computer_name
		self.logOutput.append('\nLooking for local sorting files on {}'.format(computer_name))


	def copy_data_for_d2_lims_upload(self):

		copyDataMessageBoxReply = QMessageBox.question(self.mainWin, 'Copy', 
			'Copy sorting files to E drive?\n After copying, validate before uploading', 
			QMessageBox.Ok | QMessageBox.No)
		if copyDataMessageBoxReply == QMessageBox.Ok:
			self.logOutput.append('copying local files for session {}. Running probes {}'.format(self.sessionID, self.probes_to_run))
			out = transfer_session(self.session_directory, probes_to_run=self.probes_to_run)
			for message in out:
				self.logOutput.append(message)


	def initiate_d2_upload(self):
		
		if self.acq_computer_name.upper() in acq_to_sync_comp_dict:
			exe_path = os.path.join(acq_to_sync_comp_dict[self.acq_computer_name.upper()], exe_relative_path)
			self.logOutput.append('\nFound day2 upload script here: {}'.format(exe_path))
		else:
			self.logOutput.append('\nCould not find sync computer for acq computer {}'.format(self.acq_computer_name.upper()))
			return


		if self.lims_pass is None:
			self.logOutput.append('\nPlease check lims for files before attempting upload')
			return

		elif self.lims_pass:
			self.logOutput.append('\nLims data for this session already exists. Aborting upload.')
			return

		if not self.passed_d2_local_validation:
			try:
				initialUploadBoxReply = QMessageBox.question(self.mainWin, 'Initiate LIMS upload', 
					'Session FAILED validation. Please confirm that you would like to OVERRIDE and upload session {} to LIMS'.format(self.sessionID), 
					QMessageBox.Ok | QMessageBox.No)
				if initialUploadBoxReply == QMessageBox.Ok:

					#out = subprocess.check_call(r"C:\Program Files\AIBS_MPE\createDay2\createDay2.exe"+' --sessionid '+ self.limsID)
					out = subprocess.check_call(exe_path +' --sessionid '+ self.limsID)
					if out==0:
						self.logOutput.append('\nLIMS upload successfully initated for session {}'.format(self.limsID))
				else:
					self.logOutput.append('\nAborting LIMS upload for session {}'.format(self.sessionID))
			except Exception as e:
				self.logOutput.append('\nCould not initiate d2 upload for session {}, due to error {}'
					.format(self.limsID, e))	
			self.logOutput.append('\nPlease validate local files before attempting upload')

		else:
			try:
				initialUploadBoxReply = QMessageBox.question(self.mainWin, 'Initiate LIMS upload', 
					'Please confirm that you would like to upload session {} to LIMS'.format(self.sessionID), 
					QMessageBox.Ok | QMessageBox.No)
				if initialUploadBoxReply == QMessageBox.Ok:
					#out = subprocess.check_call(r"C:\Program Files\AIBS_MPE\createDay2\createDay2.exe"+' --sessionid '+ self.limsID)
					out = subprocess.check_call(exe_path +' --sessionid '+ self.limsID)
					if out==0:
						self.logOutput.append('\nLIMS upload successfully initated for session {}'.format(self.limsID))
				else:
					self.logOutput.append('\nAborting LIMS upload for session {}'.format(self.sessionID))
			except Exception as e:
				self.logOutput.append('\nCould not initiate d2 upload for session {}, due to error {}'
					.format(self.limsID, e))	

		

	def closeEvent(self, event):
		pass

if __name__ == '__main__':
    start()


