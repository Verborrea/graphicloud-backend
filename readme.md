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
python3 generate_icons.py
uvicorn main:app --reload
```

El servidor estará disponible en: http://localhost:8000

---

Si subo 1 documento:

```
{
	"global": [
		{
			"word": "machine learning",
			"score": 0.7123
		},
		{
			"word": "neural network",
			"score": 0.6841
		}
	],
	"locals": []
}
```

Si subo 2:

```
{
	"global": [
		{
			"word": "computer vision",
			"score": 0.71
		}
	],
	"locals": [
		{
			"filename": "doc1.pdf",
			"x": 0,
			"y": 0,
			"keywords": [
				{ "word": "ai", "score": 0.52 }
			]
		},
		{
			"filename": "doc2.pdf",
			"x": 1,
			"y": 1,
			"keywords": [
				{ "word": "deep learning", "score": 0.61 }
			]
		}
	],
	"similarity": 0.7342
}
```

Y si subo 3:

```
{
	"global": [
		{
			"word": "computer vision",
			"score": 0.71
		}
	],
	"locals": [
		{
			"filename": "doc1.pdf",
			"x": 0.12,
			"y": 0.88,
			"keywords": [
				{
					"word": "cnn",
					"score": 0.53
				}
			]
		},
		{
			"filename": "doc2.pdf",
			"x": 0.67,
			"y": 0.24,
			"keywords": [
				{
					"word": "nlp",
					"score": 0.49
				}
			]
		}
	]
}
```
