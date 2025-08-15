import numpy as np
import pandas as pd
from io import StringIO
import argparse
from sklearn.cluster import DBSCAN
from scipy.stats import linregress


class AnalyzeLammpsLog:
    """
    Class for handling lammps log files.

    This class is adapted from henriasv.github.io/lammps-logfile

    :param ifile: path to lammps log file
    :type ifile: string
    """

    def __init__(self, ifile):
        # Identifiers for places in the log file
        self.start_thermo_strings = [
            'Memory usage per processor', 'Per MPI rank memory allocation'
        ]
        self.stop_thermo_strings = ['Loop time', 'ERROR']
        self.data_dict = {}
        self.keywords = []
        self.output_before_first_run = ''
        self.partial_logs = []
        self.read_file_to_dict(ifile)

    def read_file_to_dict(self, log_file):
        '''
        Store lammps log data into a dictionary

        :param ifile: path to lammps log file
        :type ifile: string
        '''

        with open(log_file, 'r') as f:
            contents = f.readlines()

        keyword_flag = False
        before_first_run_flag = True
        i = 0
        while i < len(contents):
            line = contents[i]
            if before_first_run_flag:
                self.output_before_first_run += line

            if keyword_flag:
                keywords = line.split()
                tmp_string = ''
                # Check whether any of the thermo stop strings
                # are in the present line
                while not sum(
                    [string in line
                     for string in self.stop_thermo_strings]) >= 1:
                    if '\n' in line:
                        tmp_string += line
                    i += 1
                    if i < len(contents):
                        line = contents[i]
                    else:
                        break
                partial_logcontents = pd.read_table(StringIO(tmp_string),
                                                    sep=r'\s+')

                if (self.keywords != keywords):
                    # If the log keyword changes
                    # i.e. the thermo data to be outputted chages,
                    # we flush all prevous log data.
                    # This is a limitation of this implementation.
                    self.flush_dict_and_set_new_keyword(keywords)

                self.partial_dict = {}
                for name in keywords:
                    self.data_dict[name] = np.append(self.data_dict[name],
                                                     partial_logcontents[name])
                    self.partial_dict[name] = np.append(
                        np.asarray([]), partial_logcontents[name])
                self.partial_logs.append(self.partial_dict)
                keyword_flag = False

            # Check whether the string matches
            # any of the start string identifiers
            if sum([
                    line.startswith(string)
                    for string in self.start_thermo_strings
            ]) >= 1:
                keyword_flag = True
                before_first_run_flag = False
            i += 1

    def flush_dict_and_set_new_keyword(self, keywords):
        self.data_dict = {}
        for entry in keywords:
            self.data_dict[entry] = np.asarray([])
        self.keywords = keywords

    def get(self, entry_name, run_num=-1):
        """
        Get time-series from log file by name.

        If the rows in the log file changes between runs,
        the logs are being flushed.

        :param entry_name: Name of the entry, for example 'Temp'
        :type entry_name: str
        :param run_num: Lammps simulations commonly involve
            several run-commands. Here you may choose what run
            you want the log data from. Default of :code:`-1`
            returns data from all runs concatenated
        :type run_num: int
        """

        if run_num == -1:
            if entry_name in self.data_dict.keys():
                return self.data_dict[entry_name]
            else:
                return None
        else:
            if len(self.partial_logs) > run_num:
                partial_log = self.partial_logs[run_num]
                if entry_name in partial_log.keys():
                    return partial_log[entry_name]
                else:
                    return None
            else:
                return None

    def get_keywords(self, run_num=-1):
        """Return list of available data columns in the log file."""

        if run_num == -1:
            return sorted(self.keywords)
        else:
            if len(self.partial_logs) > run_num:
                return sorted(list(self.partial_logs[run_num].keys()))
            else:
                return None

    def to_exdir_group(self, name, exdirfile):
        group = exdirfile.require_group(name)
        for i, log in enumerate(self.partial_logs):
            subgroup = group.require_group(str(i))
            for key, value in log.items():
                key = key.replace('/', '.')
                subgroup.create_dataset(key, data=value)

    def get_num_partial_logs(self):
        return len(self.partial_logs)

    @staticmethod
    def extract_msd(args):
        """
        Reads lammps log file and extract required data


        This function can be used to determine if the system
        is solid or liquid using mean square displacement analysis
        """

        parser = argparse.ArgumentParser(
            description="Read contents from lammps log files")
        parser.add_argument("log_file", type=str, help="Lammps log file")
        parser.add_argument("timestep", type=str, help="timestep")
        parser.add_argument("lattice_param", type=str, help="lattice_param")

        args = parser.parse_args(args)
        log = AnalyzeLammpsLog(args.log_file)
        timestep = float(args.timestep)
        lattice_param = float(args.lattice_param)

        msd = log.get("c_msd[4]", run_num=log.get_num_partial_logs() - 1)

        # The values are based on the metal units in LAMMPS (distance = Angs,
        # time = ps). To analyze the solid/liquid phases, slope is measured
        # only for the interval where total msd is larger than square of
        # lattice constant to make sure that the system is not in the slowly
        # increasing MSD regime.

        if msd[-1] > pow(lattice_param, 2):
            idx_above_limit = next(x[0] for x in enumerate(msd)
                                   if x[1] > pow(lattice_param, 2))
            interval = 100  # every 100 x timestep ps

            n_points = len(msd[idx_above_limit:-1])
            list_xaxis = list(range(0, interval * n_points, interval))
            list_xaxis = np.asarray(list_xaxis)
            # convert steps to time (ps)
            list_xaxis = list_xaxis * timestep

            slope, intercept, r_value, p_value, std_err = linregress(
                list_xaxis, msd[idx_above_limit:-1])
        else:
            slope = 0.0

        # In addition to having a total MSD larger than square of lattice
        # constant, the slope of MSD vs time plot should be positive to
        # make sure it is liquid phase.

        if msd[-1] > pow(lattice_param, 2) and slope - std_err > 0:
            return "liquid"
        else:
            return "solid"

    @staticmethod
    def extract_density(args):
        """

        This function can be used to determine if the system
        is solid or liquid using mean square displacement analysis
        """

        parser = argparse.ArgumentParser(
            description="Read contents from lammps log files")
        parser.add_argument("log_file", type=str, help="Lammps log file")

        args = parser.parse_args(args)
        log = AnalyzeLammpsLog(args.log_file)

        den_arr = log.get("Density", run_num=log.get_num_partial_logs() - 1)

        ave_den = np.mean(den_arr)

        return ave_den

    @staticmethod
    def extract_q(args):
        """
        Reads lammps output file and extract required data

        This function can be used to determine if the system
        is solid or liquid using q parameter
        """

        parser = argparse.ArgumentParser(
            description="Read contents from lammps log files")
        parser.add_argument("run_path",
                            type=str,
                            help="path where lammps simulation is running.")
        parser.add_argument("log_file", type=str, help="Lammps log file")
        parser.add_argument("q_file",
                            type=str,
                            help="Steinhard Q profile from lammps simulation.")
        parser.add_argument("eps_dbscan",
                            type=str,
                            help="eps parameter for dbscan")

        args = parser.parse_args(args)
        log = AnalyzeLammpsLog(args.log_file)
        run_path = args.run_path
        q = args.q_file
        eps_param = float(args.eps_dbscan)
        ave_temp = None

        with open(q) as f:
            multiple_frames = False
            count = 0
            q_z = []
            n_clusters = []
            n_atms = []
            total_atoms = []
            ave_q = []
            for line in f:
                if line.startswith("#"):
                    pass
                else:
                    if len(line.split()) == 3 and count > 0:
                        multiple_frames = True
                        if len(q_z) != 0:
                            q_z = np.reshape(q_z, (-1, 1))
                            clustering = DBSCAN(eps=eps_param,
                                                min_samples=30).fit(q_z)
                            clusters_unique = np.unique(clustering.labels_)
                            n_clusters.append(len(clusters_unique))

                        q_z = []
                        n_atms = []
                    elif len(line.split()) != 3:
                        q_z.append(float(line.split()[-1]))
                        n_atms.append(int(line.split()[2]))
                        count += 1

        if multiple_frames is False and len(q_z) != 0:
            q_z = np.reshape(q_z, (-1, 1))
            n_atms = np.reshape(n_atms, (-1, 1))
            clustering = DBSCAN(eps=eps_param, min_samples=30).fit(q_z)
            clusters_unique = np.unique(clustering.labels_)
            n_clusters.append(len(clusters_unique))

        indexes = np.unique(clustering.labels_, return_index=True)[1]
        label_unique = [clustering.labels_[index] for index in sorted(indexes)]

        for i in range(0, len(label_unique)):
            label_indexes = np.where(clustering.labels_ == label_unique[i])[0]
            n_atms_list = [n_atms[k] for k in label_indexes]
            q_list = [q_z[j] for j in label_indexes]

            ave_q.append(sum(q_list) / len(q_list))
            total_atoms.append(sum(n_atms_list))

        # Find the average temperature
        if 'NPT' in run_path:
            temp = log.get("Temp", run_num=log.get_num_partial_logs() - 1)
        else:
            temp = log.get("f_temp_avg",
                           run_num=log.get_num_partial_logs() - 1)
        ave_temp = np.mean(temp[int(len(temp) - 100):-1])
        std_temp = np.std(temp[int(len(temp) - 100):-1])

        if all(i >= 2
               for i in n_clusters[int(len(n_clusters)
                                       - 20):-1]) and len(n_clusters) > 0:

            return True, ave_temp, std_temp, label_unique, ave_q, total_atoms
        else:

            return False, ave_temp, std_temp, label_unique, ave_q, total_atoms
