"""Links de fotos definidos na mão (opcional).

Serve para dois casos: quando a Wikipédia não tem foto boa de alguém, e quando
a busca automática traz a imagem errada. O que estiver aqui tem prioridade
sobre a busca — se um ator aparece neste arquivo, o script nem consulta a
Wikipédia para ele.

Use links diretos para o arquivo de imagem (terminados em .jpg/.png), não para
a página que mostra a imagem.

    PHOTO_URLS = {
        "Selton Mello": [
            "https://upload.wikimedia.org/.../Selton_Mello.jpg",
            "https://exemplo.com/outra-foto.jpg",
        ],
    }

Depois de editar, rode de novo:

    python -m scripts.build_actors --merge
"""

PHOTO_URLS: dict[str, list[str]] = {}
