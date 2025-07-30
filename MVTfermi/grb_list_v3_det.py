import os
import numpy as np
import pandas as pd
import yaml
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from astropy.io import fits
from mvt_data_fermi import mvtfermi, format_par_as_yaml
from trigger_process import trigger_file_list,  get_dets_list
import smtplib
from email.message import EmailMessage

# ========= USER SETTINGS =========
MAX_WORKERS = 16  # You can change this to 16 if needed
BATCH_WRITE_SIZE = 16  # Number of results to write to CSV at once
DATA_PATH = '/GBMdata/triggers'
GRB_LIST_FILE = 'grb_list.txt'
TRIGGER_CONFIG_FILE = 'config_MVT_fermi.yaml'
GMAIL_FILE = 'config_mail.yaml' 
# =================================


def send_email(input='!!'):
    msg = EmailMessage()
    msg['Subject'] = 'Python Script Completed'
    msg['From'] = '2210sumaanbala@gmail.com'
    msg['To'] = 'sumanbala2210@gmail.com'
    msg.set_content(f'Hey, your script has finished running!\n{input}')

    with open(GMAIL_FILE, 'r') as f:
        config_mail = yaml.safe_load(f)

    # Use your Gmail App Password here
    gmail_user = config_mail['gmail_user']
    gmail_password = config_mail['gmail_password']

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(gmail_user, gmail_password)
        smtp.send_message(msg)

# Load trigger config
with open(TRIGGER_CONFIG_FILE, 'r') as f:
    config_template = yaml.safe_load(f)

# Read GRB trigger list
with open(GRB_LIST_FILE, 'r') as f:
    grb_list = f.read().splitlines()

# Setup output folder
script_dir = os.path.dirname(os.path.abspath(__file__))
now = datetime.now().strftime("%d_%m_%H:%M:%S")
output_dir = f'Trigger_number_vs_mvt_{now}'
output_path = os.path.join(script_dir, output_dir)
os.makedirs(output_path, exist_ok=True)

# CSV path
output_csv = f"{output_dir}.csv"
output_csv_path = os.path.join(output_path, output_csv)


# ========= CORE FUNCTIONS =========

def get_GRB_par(trigger_number, data_path):
    try:
        trigger_directory = os.path.join(data_path, "bn" + trigger_number)
        bcat_down_list_path, _ = trigger_file_list(trigger_directory, "bcat", trigger_number)
        bcat_hdu = fits.open(bcat_down_list_path[-1])
        T90 = round(float(bcat_hdu[0].header['T90']), 4)
        T50 = round(float(bcat_hdu[0].header['T50']), 4)
        PF64 = round(float(bcat_hdu[0].header['PF64']), 4)
        PFLX = round(float(bcat_hdu[0].header['PFLX']), 4)
        FLU = round(float(bcat_hdu[0].header['FLU'])*1e6, 4)
        bcat_hdu.close()
        return T90, T50, PF64, PFLX, FLU
    except Exception as e:
        print(f"Failed to get T90 for {trigger_number}: {e}")
        return None, None, None, None, None


def process_grb(trigger_number):
    config = config_template.copy()
    try:
        T90, T50, PF64, PFLX, FLU  = get_GRB_par(trigger_number, DATA_PATH)
    except:
        T90, T50, PF64, PFLX, FLU = -1  # Use -1 to indicate failure in getting T90

    
    
    config.update({
        'trigger_number': trigger_number,
        'T90': T90,
        'det_list': 'all',
        'background_intervals': [[0, 0], [0, 0]],
        'output_path': output_path,
        'bw': 0.0001,  # Set bin width to 0.0001
    })

    yaml_file_name = f'config_MVT_fermi_{trigger_number}.yaml'
    yaml_path = os.path.join(output_path, yaml_file_name)
    yaml_content = format_par_as_yaml(config, '')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(yaml_content)

    try:
        mvt_path = mvtfermi(config=yaml_path)
        with open(mvt_path, 'r') as f:
            mvt_dic = yaml.safe_load(f)

        mvt = mvt_dic['tmin']
        mvt_error = mvt_dic['dtmin']

        return {
            'trigger_number': str(trigger_number),
            'mvt_ms': round(float(mvt) * 1000, 3) if mvt else 0,
            'mvt_error_ms': round(float(mvt_error) * 1000, 3) if mvt_error else 0,
            'T90': round(float(T90), 3) if T90 else 0,
            'T50': round(float(T50), 3) if T50 else 0,
            'PF64': round(float(PF64), 3) if PF64 else 0,
            'PFLX': round(float(PFLX), 3) if PF64 else 0,
            'FLUxe6': round(float(FLU), 3) if FLU else 0,
        }

    except Exception as e:
        print(f"\nError in {trigger_number}: {e}")
        traceback.print_exc()
        return {
            'trigger_number': str(trigger_number),
            'mvt_ms': -100,
            'mvt_error_ms': -100,
            'T90': round(float(T90), 3) if T90 else 0,
            'T50': round(float(T50), 3) if T50 else 0,
            'PF64': round(float(PF64), 3) if PF64 else 0,
            'PFLX': round(float(PFLX), 3) if PF64 else 0,
            'FLUxe6': round(float(FLU), 3) if FLU else 0,
        }


# ========= MAIN PARALLEL LOGIC =========

def main():
    print(f'\nStarting MVT analysis for {len(grb_list)} GRBs using {MAX_WORKERS} workers...')
    print(f'Output directory: {output_dir}\n')

    results_batch = []
    header_written = os.path.exists(output_csv_path)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_grb, trig) for trig in grb_list]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results_batch.append(result)

            print(f"[{i}/{len(grb_list)}] Done: {result['trigger_number']}")

            # Write batch to CSV every N results
            if len(results_batch) >= BATCH_WRITE_SIZE:
                df_batch = pd.DataFrame(results_batch)
                df_batch.to_csv(output_csv_path, mode='a', index=False, header=not header_written)
                header_written = True
                results_batch = []

    # Write any remaining results
    if results_batch:
        df_batch = pd.DataFrame(results_batch)
        df_batch.to_csv(output_csv_path, mode='a', index=False, header=not header_written)
    
    send_email(input=f"Analysis completed for {len(grb_list)} GRBs! \nResults saved to {output_csv_path}")

    print(f'\nAll GRBs processed! Results saved to:\n{output_csv_path}\n')


if __name__ == '__main__':
    main()
