from treerag.models import TreeNode
from pypdf import PdfReader
from json import load, dump

def page_loader(pdf: str) -> list[str]:
    reader = PdfReader(pdf)
    pages = []
    for p in reader.pages:
        content = p.extract_text(
            # extraction_mode="layout",
            orientations=(0,)
            )
        pages.append(content)
    return pages

def extract(startpage: int, endpage: int, pages: list[str]) -> str:
    return "/n".join(pages[startpage-1:endpage])

def chunk(root: list[TreeNode], pages: list[str]) -> list[(str, str)]:
    #extract text for each node

    result = []
    for node in root:
        if len(node["children"]) != 0: 
            print(f"ANode_id: {node['node_id']}, Start_page: {node['start_page']}, End_page:{node['children'][0]['start_page']}")
            text  = extract(node["start_page"], node["children"][0]["start_page"], pages)
            result.append((node["node_id"], text))
            # input("Press Enter to continue...")
            result += chunk(node["children"], pages)    
            # print(result)
        else:
            print(f"BNode_id: {node['node_id']}, Start_page: {node['start_page']}, End_page:{node['end_page']}") 
            text = extract(node["start_page"], node["end_page"], pages)
            result.append((node["node_id"], text))
            # input("Press Enter to continue...")
            # print(result)
    
    return result

with open("sample_treeindex.json", "r") as file:
    data = load(file)

result = chunk(data["roots"], pages=page_loader("uploads\Manual_Goods_2024.pdf"))
with open("content.json", "w") as file:
    dump(dict(result), file, indent=4)