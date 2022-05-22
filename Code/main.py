# Amazon product co-purchasing network metadata
# https://snap.stanford.edu/data/amazon-meta.html

import preProcessing.preprocessing
from nltk.corpus import stopwords
from stemming.porter2 import stem
import networkx as nx
import matplotlib.pyplot as plt

# print purchased book info
def printDataOfPurchasedBook(purchasedBook,books):
    print("ASIN: ", purchasedBook)
    print("Title: ", books[purchasedBook]['Title'])
    print("SalesRank: ", books[purchasedBook]['SalesRank'])
    print("Total Reviews: ", books[purchasedBook]['TotalReviews'])
    print("Avarage Rating: ", books[purchasedBook]['AvgRating'])
    print("Degree Centrality: ", books[purchasedBook]['DegreeCentrality'])
    print("Clustering Coeff: ", books[purchasedBook]['ClusteringCoeff'])


# Read the data from the amazon-books.txt add data to books dictionary
def readData(filename,books):
    fp = open(filename, 'r', encoding='utf-8', errors='ignore')
    fp.readline()
    for line in fp:
        cell = line.split('\t')
        data = {}
        data['Id'] = cell[0].strip()
        ASIN = cell[1].strip()
        data['Title'] = cell[2].strip()
        data['Categories'] = cell[3].strip()
        data['Group'] = cell[4].strip()
        data['Copurchased'] = cell[5].strip()
        data['SalesRank'] = int(cell[6].strip())
        data['TotalReviews'] = int(cell[7].strip())
        data['AvgRating'] = float(cell[8].strip())
        data['DegreeCentrality'] = int(cell[9].strip())
        data['ClusteringCoeff'] = float(cell[10].strip())
        books[ASIN] = data
    fp.close()


# Build the depth-1 ego network graph of purchasedBook from copurchaseGraph
def printEgoNetwork(purchasedBook,copurchaseGraph):
    n = purchasedBook
    ego = nx.ego_graph(copurchaseGraph, n, radius=1)
    graph = nx.Graph(ego)
    pos = nx.layout.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=50, node_color='blue')
    nx.draw_networkx_edges(graph, pos, node_size=50, edge_cmap=plt.cm.Blues, width=2, alpha=0.1)
    ax = plt.gca()
    ax.set_axis_off()
    plt.title('Ego Network Degree 1')
    plt.show()

    #remove edges with threshold < 0.5
    threshold = 0.5
    updatedGraph = nx.Graph()
    for f,t,e in graph.edges(data=True):
        if e['weight'] >= threshold:
            updatedGraph.add_edge(f,t, weight=e['weight'])
    pos = nx.layout.spring_layout(updatedGraph)
    nx.draw_networkx_nodes(updatedGraph, pos, node_size=50, node_color='blue', label=True)
    nx.draw_networkx_edges(updatedGraph, pos, node_size=50, edge_cmap=plt.cm.Blues, width=2, alpha=0.1)
    ax = plt.gca()
    ax.set_axis_off()
    plt.title('Ego Network Degree 1 with threshold of 0.5')
    plt.figure(1)
    plt.show()
    return updatedGraph

# Display Top Five book recommendations from all the purchasedBookNeighbours
def printFiveRecomandations(books,purchasedBookNeighbours):
    data = []
    for asin in purchasedBookNeighbours:
        ASIN = asin
        Title = books[asin]['Title']
        SalesRank = books[asin]['SalesRank']
        TotalReviews = books[asin]['TotalReviews']
        AvgRating = books[asin]['AvgRating']
        DegreeCentrality = books[asin]['DegreeCentrality']
        ClusteringCoeff = books[asin]['ClusteringCoeff']
        data.append((ASIN, Title, SalesRank, TotalReviews, AvgRating, DegreeCentrality, ClusteringCoeff))
        
    # Sort the top five nodes in purchasedBookNeighbour by Average Rating and TotalReviews
    topRecomandedBooks = sorted(data, key=lambda x: (x[4], x[3]), reverse=True)[:5]

    # Print Top 5 Recommendations
    print("------------------------------------------------------------------------------------")
    print('Top 5 Recommendations for the book:')
    print('------------------------------------------------------------------------------------')
    print('ASIN\t', 'Title\t', 'SalesRank\t', 'TotalReviews\t', 'AvgRating\t', 'DegreeCentrality\t', 'ClusteringCoeff')
    for asin in topRecomandedBooks:
        print(asin)
    print('------------------------------------------------------------------------------------')


if __name__ == "__main__":
    books = {}
    readData("amazon-books.txt",books) 

    # Assign a weighted graph from edgelist data
    fp = open("amazon-books-copurchase.edgelist", "rb")
    copurchaseGraph = nx.read_weighted_edgelist(fp)
    fp.close()

    print("\n------------------------------------------------------------------------------------")
    print("Purchased Book Info:")
    print("------------------------------------------------------------------------------------")
    # User purchased book which asin is give below:
    purchasedBook = '0842328327'

    printDataOfPurchasedBook(purchasedBook,books)

    updatedGraph = printEgoNetwork(purchasedBook,copurchaseGraph)

    # Get the list of nodes that are connected to the purchasedBook
    purchasedBookNeighbours = updatedGraph.neighbors(purchasedBook)

    printFiveRecomandations(books,purchasedBookNeighbours)