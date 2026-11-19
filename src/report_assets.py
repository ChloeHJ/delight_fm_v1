"""Make the HTML report portable by embedding each local image."""
import base64
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote


def embed_images(page,root):
    root=Path(root)
    cache={}
    def replace(match):
        value=match.group(3)
        if value.startswith('data:'):return match.group(0)
        if '://' in value:raise ValueError('Report image must be local: '+value)
        if value not in cache:
            path=root/unquote(value)
            mime=mimetypes.guess_type(path.name)[0]
            if not mime or not mime.startswith('image/'):raise ValueError('Invalid image type: '+str(path))
            cache[value]='data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
        return match.group(1)+match.group(2)+cache[value]+match.group(2)
    return re.sub(r'(<img\b[^>]*?\bsrc=)([\"\x27])(.*?)\2',replace,page,flags=re.I|re.S)
