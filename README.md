**Step 1: Install Dependencies**
Install the required Python packages: pip install -r requirements.txt

**Step 2: Run test_deepeval.py**
Execute the following command: deepeval test run test_deepeval.py

**Step 3: Modify test_deepeval.py**
Open the test_deepeval.py file and locate the test_hallucination() function. Add a parameter chatbot to the function signature.

    #Before modification
    def test_hallucination():

    #After modification
    def test_hallucination(chatbot):

Save the file after making the modification.

**Step 4: Run Flask Server**
Start the Flask server: python flask_server.py

**Step 5: Test with Postman**
Open Postman and send a POST request to the following URL : http://127.0.0.1:5000/info