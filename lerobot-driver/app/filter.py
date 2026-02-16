

if __name__ == "__main__":
    import rerun as rr
    recording = rr.dataframe.load_recording("/home/phillip/Downloads/data.rrd")
    view = recording.view(index="log_time", contents="*")
    batches = view.select()
