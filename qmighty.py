#!/usr/bin/env python
import subprocess, threading, time, datetime, os
import lammps_logfile


class Process:
    def __init__(self, path:str, name:str, outfile_name:str="qmighty-out.txt"):
        self.path = path
        self.name = name
        self.outfile_name = outfile_name
        self.elapsed_time = 0.0
        self.remaining_time = 0.0

    def set_command(self, command:str):
        self.command = command

    def spawn_thread(self):
        outfile = open(os.path.join(self.path, self.outfile_name), "w")
        self.process = subprocess.Popen(self.command,
                         shell=False,
                         stdin=subprocess.PIPE, stdout=outfile,
                         stderr=subprocess.PIPE)

        self.thread = threading.Thread(target=self.print_errors,args=(self.process,))
        self.thread.daemon = True
        self.thread.start()

    def print_errors(self, process):
        for line in process.stderr:
            print("STDERR: ", line.decode("utf-8"))

    def update_timings(self, t0):
        t1 = time.time()
        self.elapsed_time = t1 - t0

        try:
            log = lammps_logfile.File(os.path.join(self.path, self.outfile_name))
            remaining_time = log.get('CPULeft', run_num=-1)
        except:
            remaining_time = None
        if remaining_time is None:
            self.remaining_time = 0.0
        else:
            self.remaining_time = round(remaining_time[-1])



class Qmighty:

    def __init__(self, root_dir:str, sim_dir_list:list=None, update_frequency:float=0.5, add_suffixes:list=None):
        self.root_dir = root_dir
        self.update_frequency = update_frequency
        self.accepted_suffixes = [".lmp", ".in"]
        if add_suffixes is not None:
            self.add_suffix(add_suffixes)

        if sim_dir_list is None:
            self.sim_dir_list = []
            dirs = os.listdir(self.root_dir)
            for dir in dirs:
                if os.path.isdir(os.path.join(self.root_dir, dir)):
                    self.sim_dir_list.append(dir)
        else:
            self.sim_dir_list = sim_dir_list

    def add_suffix(self, suffixes:list):
        for suffix in suffixes:
            if suffix not in self.accepted_suffixes:
                self.accepted_suffixes.append(suffix)

    def sanity_check(self, sim_path:str):
        filenames = os.listdir(sim_path)
        for filename in filenames:
            for suffix in self.accepted_suffixes:
                if suffix in filename:
                    return True

        return False



    def init_processes(self, computer:str, input_filename:str):
        if computer != 'CPU' and computer != 'GPU':
            raise ValueError(f"computer={computer}, not 'CPU' or 'GPU'")

        self.queue = {}
        procID = 100
        self.num_procs = 0
        for sim_dir in self.sim_dir_list:
            sim_path = os.path.join(self.root_dir, sim_dir)

            if not self.sanity_check(sim_path):
                continue

            if computer=='CPU':
                command = f'mpirun -np 4 -wdir {sim_path} lmp -in {input_filename}'.split()
            if computer=='GPU':
                command = f"mpirun -n 1 -wdir {sim_path} lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in {input_filename}".split()

            proc = Process(path = sim_path, name = sim_dir)
            proc.set_command(command)

            self.queue[procID] = {"ID": procID,
                                  "Process": proc}

            procID += 1
            self.num_procs += 1


    def run_process_loop(self):
        self.id_currently_running = 0
        self.clear_n_lines()
        t0_main = time.time()
        for current_proc_ID, value in self.queue.items():
            self.id_currently_running = current_proc_ID
            process = value['Process']
            process.spawn_thread()
            t0_proc = time.time()
            while process.thread.is_alive():
                process.update_timings(t0_proc)
                self.print_progress()
                time.sleep(self.update_frequency)

            process.complete = True
            if process.thread.is_alive():
                process.thread.join()
        tfinal_main = time.time() - t0_main
        if self.num_procs == 0:
            print("Found no LAMMPS input files. Pro-tip: check input filename formatting.")
        else:
            print(f"\n{self.num_procs} runs completed. Run time: {tfinal_main/3600:.2f} hrs")


    @staticmethod
    def clear_n_lines(n:int=None):

        #clear entire screen
        if n is None:
            print("\033[1J\033[2J\033[3J\033[f", end="")
        #clear n lines from (0,0) (top left)
        else:
            print(f"\033[{n};0f\033[1J", end="")
            print("\033[f", end="")

    def _issue_commands(self):
        self.user_input = None
        t = threading.Thread(target=self.get_input)
        t.daemon = True
        t.start()

        t.join(timeout=self._command_input_timeout)
        return self.user_input

    def print_progress(self):
        head = ["Job ID", "Name", "Elapsed Time", "Remaining Time"]
        self.clear_n_lines(self.num_procs + 5)
        print(f"{head[0]} \t\t {head[1]} \t\t\t {head[2]} \t\t {head[3]}")
        for id, value in self.queue.items():
            process = value['Process']
            remaining = datetime.timedelta(seconds=round(process.remaining_time))
            elapsed = datetime.timedelta(seconds=round(process.elapsed_time))
            print(f"{id} \t\t {process.name} \t {elapsed} \t\t {remaining}")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser("qmighty.py")

    parser.add_argument(
        "-d", "--root_dir", dest="root_dir", metavar="SIMULATION(S)_ROOT_DIRECTORY",
        default="./",
        help="Path to root directory containing simulations."
        " Must also be set in the case of a single simulation."
        )
    parser.add_argument(
        "-s", "--simulation_dir", dest="simulation_dir", nargs="+", metavar="SIMULATION_DIR",
        default=None,
        help="(Optional) Name(s) of directory containing simulation files.\n"
             "If not set, all directories contained within root_dir are assumed to contain simulations."
        )
    parser.add_argument(
        "-i", "--input_format", dest="input_format", metavar="INPUT_FILE_FORMAT",
        default="run.lmp",
        help="Format of LAMMPS input script. Assumed to be equal in all simulations. \n"
             "(Differing input script names currently not supported.)"
        )
    parser.add_argument(
        "-c", "--computer", default="GPU", metavar="COMPUTER",
        choices=["CPU", "GPU"],
        help="Set as 'CPU' or 'GPU'."
        )
    parser.add_argument(
        "-u", "--update", default="1", type=float, dest="update", metavar="TIMER_UPDATE_REQUENCY",
        help="Set frequency (seconds) of progress monitor updates."
    )

    args = parser.parse_args()

    queue = Qmighty(args.root_dir, args.simulation_dir, args.update, args.input_format)
    queue.init_processes(args.computer, args.input_format)
    queue.run_process_loop()
















#
