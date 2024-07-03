#### Step 1: Install the virtualenv package

First, you need to install the `virtualenv` package. You can do this using pip:

```bash
pip install virtualenv
```

#### Step 2: Create a Virtual Environment

```bash
virtualenv venv
```

#### Step 3: Activate the Virtual Environment

- On Windows, run:

```bash
venv\Scripts\activate
```

- On Unix or MacOS, run:

```bash
source venv/bin/activate
```

#### Step 4: Deactivate the Virtual Environment
```bash
deactivate
```
#### Step 5: Download package and library
```
pip install -r requirements.txt
```
#### Step 6: Generate adversarial examples
At line 119 of generator/payload_gen.py, change to your desire PPO agents (ppo_model_cnn.zip, ppo_model_lstm.zip, ppo_model_mlp.zip)
```
python3 generator/payload_gen.py
```