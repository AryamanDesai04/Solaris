ALLOWED_EXTENSIONS = {'.dcm'}

def allowed_file(filename):
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return f'.{ext}' in ALLOWED_EXTENSIONS
