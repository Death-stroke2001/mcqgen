from setuptools import find_packages,setup

setup(
    name='mcqgenrator',
    version='0.0.1',
    author='vineet dhokare',
    author_email='vpd2001@gmail.com',
    install_requires=["openai","huggingface-hub", "langchain","langchain_community", "streamlit","python-dotenv","PyPDF2"],
    packages=find_packages()
)