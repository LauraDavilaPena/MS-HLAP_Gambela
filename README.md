# Mixed-integer optimization for the MS-HLAP

This repository contains a mixed-integer optimization (MIO) model for solving a multi-service hierarchical location-allocation problem (MS-HLAP). The MS-HLAP is designed to improve access to health facilities within refugee camps in the Gambela region, Ethiopia, by considering both the hierarchical structure of the system and the variety of healthcare services provided.

## Key Features:

- A **novel MIO model** tailored to improve healthcare access.
- Focus on the **hierarchical nature** of healthcare distribution in refugee camps.
- Utilizes **simulated data** based on insights from an NGO working in the Gambela region.
  

## Requirements:
This model has been implemented using Python **3.12.7** and has been tested on **Windows 11** laptops. To run the code, you'll need to have the following installed:

- Python 3.12.7 (or later)
- Relevant Python packages (e.g., `numpy`, `pandas`, `matplotlib`, `itertools`, `scipy`, `pyomo`, etc.)

### Installing dependencies
You can install all required dependencies through the provided Conda environment, or you can install packages manually using pip for individual needs.

- If you're using **Conda**, use the `environment.yml` file.
- If you prefer **pip**, you can install the necessary packages using:

```bash
pip install numpy pandas matplotlib itertools scipy pyomo

## Setting Up the Environment: 

To set up the environment and install all required dependencies, you can use the provided `environment.yml` file.

### Steps (use the Anaconda Prompt terminal):

1. **Clone the repository**:
   git clone https://github.com/LauraDavilaPena/MS-HLAM_Gambela.git
   cd MS-HLAM_Gambela

2. **Create the Conda environment from the `environment.yml` file**:
   conda env create -f environment.yml

3. **Activate the environment**:
   conda activate Gambella_refcamps

4. **Run the model**:
- If you're using **Jupyter Notebooks**, start Jupyter with:
  jupyter notebook

- If you prefer running the code as a **Python script**, you can execute it directly from the terminal once the environment is activated:
  python mio_ms-hlap.py


