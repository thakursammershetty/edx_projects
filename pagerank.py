import os
import re
import sys
import random

DAMPING = 0.85
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)
    print("PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

def crawl(directory):
    pages = dict()

    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = set()
            for link in re.findall(r'href="(.*?)"', contents):
                if link.startswith("#"):
                    continue
                if link.startswith("mailto:"):
                    continue
                links.add(link)
            pages[filename] = set(links) - {filename}

    for filename in pages:
        pages[filename] = {link for link in pages[filename] if link in pages}

    return pages

def transition_model(corpus, page, damping_factor):
    distribution = {}
    pages = corpus.keys()
    num_pages = len(pages)

    links = corpus[page]
    num_links = len(links)

    if num_links == 0:
        for p in pages:
            distribution[p] = 1 / num_pages
    else:
        for p in pages:
            distribution[p] = (1 - damping_factor) / num_pages
            if p in links:
                distribution[p] += damping_factor / num_links

    return distribution

def sample_pagerank(corpus, damping_factor, n):
    pagerank = {page: 0 for page in corpus}
    sample = random.choice(list(corpus.keys()))

    for _ in range(n):
        pagerank[sample] += 1
        distribution = transition_model(corpus, sample, damping_factor)
        sample = random.choices(list(distribution.keys()), weights=distribution.values(), k=1)[0]

    pagerank = {page: rank / n for page, rank in pagerank.items()}
    return pagerank

def iterate_pagerank(corpus, damping_factor):
    num_pages = len(corpus)
    pagerank = {page: 1 / num_pages for page in corpus}
    new_pagerank = pagerank.copy()

    while True:
        for page in corpus:
            total = 0
            for linking_page in corpus:
                if page in corpus[linking_page]:
                    total += pagerank[linking_page] / len(corpus[linking_page])
                if not corpus[linking_page]:
                    total += pagerank[linking_page] / num_pages

            new_pagerank[page] = (1 - damping_factor) / num_pages + damping_factor * total

        if max(abs(new_pagerank[page] - pagerank[page]) for page in corpus) < 0.001:
            break

        pagerank = new_pagerank.copy()

    return pagerank

if __name__ == "__main__":
    main()

