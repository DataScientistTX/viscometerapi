# A Dashboard to Calculate Rheological Properties from any API Viscometer

## Installation

In the  terminal from the home directory, use the command git clone, then paste the link from your clipboard, or copy the command and link from below:

```bash
git clone https://github.com/sercangul/viscometerapi.git
```

Change directories to the new ~/viscometerapi directory:

```bash
cd viscometerapi
```

To ensure that your master branch is up-to-date, use the pull command:

```bash
git pull https://github.com/sercangul/viscometerapi.git
```

Install required python packages using requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

Change directories to the new ~/viscometerapi directory:

```bash
cd viscometerapi
```

Run the script using Streamlit:

```bash
streamlit run app.py
```

The same app is also deployed to Heroku: http://viscometerapi.herokuapp.com/

Enter your dial readings obtained from an API viscometer using the template excel sheet and investigate the rheological behavior of your fluid with curve fit results provided for Bingham Plastic, Power-Law and Yield-Power Law rheological models.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)
