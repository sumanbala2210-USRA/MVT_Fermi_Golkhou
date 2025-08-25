import argparse
import os

def main(trigger_number, nai='n1'):
    """
    Main function to execute the getGBMdata script with command line arguments.
    """
    path = os.path.join(os.getcwd(), f"bn{trigger_number}")
    # Example command to run the script
    command = f"getGBMdata -dest {path}  bn {trigger_number} --data cspec_rsp2 tte --nai {nai} -protocol a"
    #command = f"getGBMdata -dest {path}  bn {trigger_number} --data tte --nai {nai} -protocol a"

    # Execute the command
    os.system(command)


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MVT data from different detector sets using specified arguments.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-bn', type=str, help='The trigger number to process.')
    #parser.add_argument('-data_type', type=str, default='rsp2', choices=['cspec_rsp2', 'tte'],
    #                    help='Type of data to retrieve (default: rsp2).')
    parser.add_argument('-nai', type=str, default='n1',
                        help='Nai detector to use (default: n1).')
    args = parser.parse_args()
    main(args.bn, args.nai)
"""
if __name__ == "__main__":
    trigger_info = [
    {'trigger': '250709653', 'det': '6', 'angle': '10.73'}, #10
    {'trigger': '250709653', 'det': '3', 'angle': '39.2'}, #40
    {'trigger': '250709653', 'det': '9', 'angle': '59.42'}, #60
    {'trigger': '250709653', 'det': '1', 'angle': '89.63'}, #90
    {'trigger': '250709653', 'det': '2', 'angle': '129.77'}, #130
    {'trigger': '250717158', 'det': '3', 'angle': '30.38'}, #30
    {'trigger': '250717158', 'det': '0', 'angle': '72.9'}, #70
    {'trigger': '250717158', 'det': '6', 'angle': '50.41'}, #50
    {'trigger': '250717158', 'det': '9', 'angle': '99.28'}, #100
    {'trigger': '250723551', 'det': '1', 'angle': '81.81'}, #80
    {'trigger': '250723551', 'det': '3', 'angle': '22.82'}, #20
    {'trigger': '250723551', 'det': '2', 'angle': '122.52'}, #120
    {'trigger': '250723551', 'det': 'a', 'angle': '141.17'}, #140
]
    for trigger in trigger_info:
        print(f"Processing trigger {trigger['trigger']} with detector {trigger['det']}")
        main(trigger['trigger'], trigger['det'])