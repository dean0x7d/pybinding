import datetime
import io
import os
import sys


def percentage(template="{:3.0%}"):
    def widget(pbar):
        return template.format(pbar.percent)
    return widget


def bar(marker='/', left='[', right=']', fill=' '):
    def widget(pbar, width):
        width -= len(left) + len(right)
        marked = marker * int(pbar.percent * width)
        return "{}{}{}".format(left, marked.ljust(width, fill), right)
    return widget


def elapsed(template="Elapsed: {}"):
    def widget(pbar):
        return template.format(datetime.timedelta(seconds=int(pbar.elapsed_seconds)))
    return widget


def eta(template="ETA: {}"):
    def widget(pbar):
        if pbar.value == 0:
            return template.format("--:--:--")
        else:
            remaining = pbar.elapsed_seconds * (1 / pbar.percent - 1)
            return template.format(datetime.timedelta(seconds=int(remaining)))
    return widget


class StdCapture:
    def __init__(self, stream_name='stdout'):
        self.stream_name = stream_name
        self._old_stream = getattr(sys, stream_name)

    def start(self):
        self._old_stream = getattr(sys, self.stream_name)
        setattr(sys, self.stream_name, io.StringIO())

    def get(self):
        output = getattr(sys, self.stream_name).getvalue()
        setattr(sys, self.stream_name, io.StringIO())
        return output

    def stop(self):
        setattr(sys, self.stream_name, self._old_stream)


class StreamOutput:
    def __init__(self, stream):
        self.stream = stream
        self.width = 80

    def start(self, width):
        self.width = width

    def clear_pbar(self):
        self.stream.write("\r" + " " * self.width + "\r")

    def write(self, line):
        self.stream.write(line)
        self.stream.flush()

    def write_pbar(self, line):
        self.stream.write('\r' + line)
        self.stream.flush()

    def stop(self):
        self.stream.write("\n")


class FileOutput:
    def __init__(self, filename):
        self.file = open(filename, 'w')

    def start(self, width):
        self.file.seek(0, 0)
        self.file.write(' ' * width + '\n')
        self.file.flush()
        self.file.seek(0, os.SEEK_END)

    def clear_pbar(self):
        pass

    def write(self, line):
        self.file.write(line)

    def write_pbar(self, line):
        self.file.seek(0, 0)
        self.file.write(line + '\n')
        self.file.flush()
        self.file.seek(0, os.SEEK_END)

    def stop(self):
        self.file.close()


class ProgressBar:
    def __init__(self, size, widgets=None, width=80, stream=sys.stdout, filename=""):
        self.size = size
        if widgets:
            self.widgets = widgets
        else:
            self.widgets = self.default_widgets()

        self.width = width
        self.outputs = []
        if stream:
            self.outputs.append(StreamOutput(stream))
        if filename:
            self.outputs.append(FileOutput(filename))

        self.captures = [StdCapture('stdout'), StdCapture('stderr')]

        self.value = 0
        self.running = False
        self.start_time = datetime.datetime.now()
        self.last_update_time = self.start_time

    @staticmethod
    def default_widgets():
        return ['Progress ', percentage(), ' ', bar(), ' ', elapsed(), ' / ', eta()]

    @property
    def percent(self):
        return self.value / self.size

    @property
    def elapsed_seconds(self):
        return (self.last_update_time - self.start_time).total_seconds()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def __iadd__(self, value):
        self.update(self.value + value)
        return self

    def _make_line(self):
        def format_fixed_size(widget):
            try:
                return widget(self)
            except TypeError:
                return widget

        semi_formatted_widgets = [format_fixed_size(w) for w in self.widgets]
        num_remaining = sum(callable(w) for w in semi_formatted_widgets)
        width_formatted = sum(len(w) if not callable(w) else 0 for w in semi_formatted_widgets)
        width_remaining = self.width - width_formatted

        def format_variable_size(widget):
            if callable(widget):
                return widget(self, int(width_remaining / num_remaining))
            else:
                return widget

        formatted_widgets = (format_variable_size(w) for w in semi_formatted_widgets)
        return "".join(formatted_widgets).ljust(self.width)

    def start(self):
        self.running = True
        self.value = 0

        for capture in self.captures:
            capture.start()
        for output in self.outputs:
            output.start(self.width)

        self.start_time = datetime.datetime.now()
        self.last_update_time = self.start_time
        self.refresh()

    def update(self, value):
        if not self.running:
            self.start()

        if self.value == value:
            return

        self.value = value
        self.last_update_time = datetime.datetime.now()
        self.refresh()

    def refresh(self):
        for output in self.outputs:
            output.clear_pbar()

        lines = [c.get() for c in self.captures]
        for output in self.outputs:
            for line in lines:
                output.write(line)

        for output in self.outputs:
            output.write_pbar(self._make_line())

    def stop(self):
        if not self.running:
            return

        self.refresh()

        for capture in self.captures:
            capture.stop()
        for output in self.outputs:
            output.stop()

        self.running = False

    def finish(self):
        if not self.running:
            return

        self.update(self.size)
        self.stop()
