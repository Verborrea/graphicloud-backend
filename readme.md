# Graphicloud Backend

Para setear el proyecto:

1. Activar el `venv`:

```
python -m venv venv

# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate
```

2. Instalar las dependencias mediante el archivo `requirements`

```
pip install requirements.txt
```

Para correrlo:

```
uvicorn main:app --reload
```

El servidor estará disponible en: http://localhost:8000
