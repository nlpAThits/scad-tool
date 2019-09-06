# scad-tool
Code and resources for the Author Name Disambiguation (AND) tool developed in the Scalable Author Disambiguation (SCAD) project

** Currently under construction for TPDL 2019 ***

<h3>Setup</h3>

<p>

```shell
$ mkdir scad
$ cd scad
$ git clone https://github.com/nlpAThits/scad-tool.git
$ cd scad-tool
$ conda create --name scad-env python=3.6
$ source activate scad-env
$ pip install -r scad-requirements-wo-wombat.txt 
$ git clone https://github.com/nlpAThits/WOMBAT.git
$ pip install WOMBAT/.
$ unzip 'resources/wombat/*.zip'
```

</p>

<h3>Starting the SCAD server</h3>

<p>
  
```shell
$ python scad-server/app.py localhost 50001 &

```  
</p>
This will start the server in the background and return a PID that can be used in 

```shell
$ kill PID

```  
to stop the server.



