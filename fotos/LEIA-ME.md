# Suas fotos entram aqui

Jogue fotos nesta pasta e rode:

```bash
python -m scripts.build_actors --photos-dir --merge
```

Cada pessoa vira uma entrada na base. O nome que aparece no resultado é o
nome da pasta (ou do arquivo) — nada de editar código.

## Dois jeitos de organizar

**Uma foto só, arquivo solto** — o nome do arquivo é o nome da pessoa:

```
fotos/
  Tom Hanks.jpg
  Fernanda Torres.jpg
```

**Várias fotos, uma pasta por pessoa** — melhor resultado, porque o app tira a
média dos rostos:

```
fotos/
  Tom Hanks/
    1.jpg
    2.jpg
  Fernanda Torres/
    premiere.jpg
    entrevista.png
```

Dá para misturar os dois na mesma pasta.

## Dicas

- 3 a 6 fotos por pessoa é o ponto ideal. Uma só já funciona.
- Rosto de frente, bem iluminado e sem óculos escuros. Foto onde a pessoa
  aparece de lado ou pequena no fundo é descartada em silêncio.
- Se a foto tiver mais de um rosto, o app usa o maior.
- `--merge` é o que faz a base crescer aos poucos: sem ele, a base é refeita
  do zero e quem não está na pasta some.
- Aceita `.jpg`, `.jpeg` e `.png`.
