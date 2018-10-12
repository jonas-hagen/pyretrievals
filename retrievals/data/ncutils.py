"""
Some tools to call binaries to deal with NetCDF files.
"""
import os
import subprocess


def concat(files, out_file):
    """
    Concat multiple files along the unlimited dimensions to one NetCDF file
    using the ncrcat command line utility.

    :param files: List of file names
    :param out_file: File to output
    :return: Return code of ncrcat
    """
    # TODO: Deal with long file lists by writing to stdin of ncrcat
    if not files:
        raise ValueError('List of files to concat is empty.')
    return subprocess.call(['ncrcat', '-h'] + files + ['-O', out_file])


def append(in_file, out_file):
    """
    Append a NetCDF file to another. This is most often used to concat
    files with disjoint groups.

    :param in_file: File to read.
    :param out_file: File to append to.
    :return: Return code of ncks
    """
    return subprocess.call(['ncks', '-h', '-A', in_file, out_file])


def which(program):
    # thanks to Jay: https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def check_prerequisites():
    required = ['ncrcat', 'ncks']
    missing = list()
    for exe in required:
        if which(exe) is None:
            missing.append(exe)

    if missing:
        raise FileNotFoundError('Missing executables ' + ', '.join(missing) + '. Please install them.')


check_prerequisites()