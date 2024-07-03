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
#### Step 6: Train PPO agent
```
python3 PPO.py
```
#### Step 6: Generate adversarial examples.
```
python3 xss_mutated.py
```