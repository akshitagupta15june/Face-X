using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
 
public class HeadTracking : MonoBehaviour
{
    public UDPReceive uDPReceive;
    public GameObject cameraObject;
    List<float> xList = new List<float>();
    List<float> yList = new List<float>();
 
 
    // Start is called before the first frame update
    void Start()
    {
        
    }
 
    // Update is called once per frame
    void Update()
    {
        string data = uDPReceive.data;
        data = data.Remove(0, 1);
        data = data.Remove(data.Length-1, 1);
 
        string[] points = data.Split(',');
 
        print(points[0]);
 
        float x = (float.Parse(points[0])-320) / 100;
        float y = (float.Parse(points[1])-240) / 100;
        xList.Add(x);
        yList.Add(y);
 
        if (xList.Count > 50) { xList.RemoveAt(0); }
        if (yList.Count > 50) { yList.RemoveAt(0); }
 
        float xAverage = Queryable.Average(xList.AsQueryable());
        float yAverage = Queryable.Average(yList.AsQueryable());
 
 
        Vector3 cameraPos = cameraObject.transform.localPosition;
        Vector3 cameraRot = cameraObject.transform.localPosition;
 
 
        cameraObject.transform.localPosition = new Vector3(-26.75f- xAverage, 2.9f- yAverage, cameraPos.z);
        cameraObject.transform.localEulerAngles = new Vector3(18.76f- yAverage * 10, xAverage * 10 , 0);
 
    }
}