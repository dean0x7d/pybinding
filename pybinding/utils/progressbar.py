from progressbar import *


class CustomETA(Timer):
    def update(self, pbar):
        if pbar.currval == 0:
            return self.format % '--:--:--'
        else:
            elapsed = pbar.seconds_elapsed
            eta = elapsed * pbar.maxval / pbar.currval - elapsed
            return self.format % self.format_time(eta)


class Range(ProgressBar):
    def __init__(self, maxval, file_name="", fd=sys.stdout, custom_widgets=None, term_width=80):
        default_widgets = ['Progress ', Percentage(), ' ', Bar('/', '[', ']'),
                           Timer(' Elapsed: %s /'), CustomETA(' ETA: %s')]

        super().__init__(widgets=custom_widgets if custom_widgets else default_widgets,
                         maxval=maxval, term_width=term_width, fd=fd, redirect_stdout=True)

        if file_name:
            self._file = open(file_name, 'w+')
        else:
            self._file = None

    def update(self, value=None):
        """ Update stdout and file """
        from io import StringIO

        if value is not None and value is not UnknownLength:
            if (self.maxval is not UnknownLength
                    and not 0 <= value <= self.maxval
                    and not value < self.currval):

                raise ValueError('Value out of range')

            self.currval = value

        if self.start_time is None:
            self.start()
            self.update(value)
        if not self._need_update():
            return

        if self.redirect_stderr and sys.stderr.tell():
            if self.fd:
                self.fd.write('\r' + ' ' * self.term_width + '\r')
            last_line = sys.stderr.getvalue()
            self._stderr.write(last_line)
            self._stderr.flush()
            sys.stderr = StringIO()
            if self._file:  # write stderr to file
                self._file.write(last_line)

        if self.redirect_stdout and sys.stdout.tell():
            if self.fd:
                self.fd.write('\r' + ' ' * self.term_width + '\r')
            last_line = sys.stdout.getvalue()
            self._stdout.write(last_line)
            self._stdout.flush()
            sys.stdout = StringIO()
            if self._file:  # write stdout to file
                self._file.write(last_line)

        now = time.time()
        self.seconds_elapsed = now - self.start_time
        self.next_update = self.currval + self.update_interval
        if self.fd:
            self.fd.write('\r' + self._format_line())

        # write progress to file
        if self._file:
            self._file.seek(0, 0)
            self._file.write(' ' * self.term_width)
            self._file.seek(0, 0)
            self._file.write(self._format_line() + '\n')
            self._file.flush()
            self._file.seek(0, os.SEEK_END)

        self.last_update_time = now

    def finish(self):
        self.finished = True
        self.update(self.maxval)
        if self.fd:
            self.fd.write('\n')
        if self.signal_set:
            signal.signal(signal.SIGWINCH, signal.SIG_DFL)

        if self.redirect_stderr:
            self._stderr.write(sys.stderr.getvalue())
            sys.stderr = self._stderr

        if self.redirect_stdout:
            self._stdout.write(sys.stdout.getvalue())
            sys.stdout = self._stdout
