using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Move_Enemy : MonoBehaviour
{
    private Transform t;
    private bool change;
    // Start is called before the first frame update
    void Start()
    {
       t =  gameObject.GetComponent<Transform>();
        change = true;
    }

    // Update is called once per frame
    void Update()
    {
        if (change == true)
        {
            t.position = t.position + t.forward * 2.0f * Time.deltaTime;
        }
        else
        {
            t.position = t.position + t.forward * -2.0f * Time.deltaTime;
        }

    }

    void OnCollisionEnter(Collision collision)
    {


        if (collision.gameObject.CompareTag("Wall"))
        {
            if (change == true)
            {
                change = false;
            }
            else
            {
                change = true;
            }

        } 

        if (collision.gameObject.CompareTag("Bullet"))
         {
                
                Destroy(collision.gameObject);
               // Destroy(gameObject);
            GameObject wall = GameObject.FindWithTag("MovableWall");//.GetComponent<GameObject>();
            wall.SetActive(false);
           // wall.GetComponent<Collider>().enabled = false;
                

         }
            


        
    }



}
