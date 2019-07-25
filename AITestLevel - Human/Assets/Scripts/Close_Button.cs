using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Close_Button : MonoBehaviour
{
    public Animator Anim;
    public GameObject Door;
    public GameObject player;

    private FPS_Camera script;
    private Collider DoorCollider;

    // Start is called before the first frame update
    void Start()
    {
        DoorCollider = Door.GetComponent<Collider>();
        script = player.GetComponent<FPS_Camera>();
    }

    // Update is called once per frame
    void Update()
    {

    }

    void OnCollisionEnter(Collision collision)
    {


        if (collision.gameObject.CompareTag("Bullet"))
        {
            Anim.ResetTrigger("character_nearby");
            DoorCollider.enabled = true;
            script.WriteString(System.DateTime.Now.ToString("yyyy/MM/dd hh:mm:ss") + "Door close checkpoint");
            Destroy(collision.gameObject);
            //RbPlayer.velocity = Vector3.zero;

        }
    }
}
